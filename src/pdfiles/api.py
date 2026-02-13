import io
import logging
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pdfiles.config import Config
from pdfiles.librarian import LibrarianDB
from pdfiles.renderer import render_page

logger = logging.getLogger(__name__)

# Global singletons set during lifespan
_searcher = None
_cfg = None

# Indexing state
_index_state = {
    "running": False,
    "stage": "",
    "scanned_pdfs": 0,
    "scanned_pages": 0,
    "classified_pages": 0,
    "indexed": 0,
    "total": 0,
    "visual_pages": 0,
    "errors": 0,
    "error_message": None,
    "scan_complete": False,
    "classify_complete": False,
}
_index_lock = threading.Lock()


def _resolve_pdf_path(stored_path: str) -> Path:
    """Resolve a stored PDF path using Config.resolve_pdf_path."""
    return _cfg.resolve_pdf_path(stored_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher, _cfg
    from pdfiles.searcher import Searcher

    _cfg = Config()
    logger.info("Loading search model on %s...", _cfg.device)
    _searcher = Searcher(_cfg)
    logger.info("Model loaded, API ready")
    yield
    _searcher = None
    _cfg = None


app = FastAPI(title="PDfiles API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    from pdfiles.qdrant_store import QdrantStore

    store = QdrantStore(_cfg)
    qdrant_connected = False
    count = 0
    try:
        # Check connection first (works even without a collection)
        store.client.get_collections()
        qdrant_connected = True
        try:
            count = store.count()
        except Exception:
            pass  # Collection doesn't exist yet
    except Exception:
        pass
    has_gpu = _cfg.device.startswith("cuda")
    data_available = _cfg.data_root.exists() and _cfg.images_root.exists()
    return {
        "collection": _cfg.collection_name,
        "indexed_pages": count,
        "gpu": has_gpu,
        "can_index": has_gpu and _cfg.admin_mode,
        "admin_mode": _cfg.admin_mode,
        "qdrant_connected": qdrant_connected,
        "data_available": data_available,
    }


@app.post("/api/index")
def start_indexing():
    if not _cfg.admin_mode:
        raise HTTPException(status_code=403, detail="Admin mode is disabled")
    if not _cfg.device.startswith("cuda"):
        raise HTTPException(status_code=400, detail="Indexing requires a GPU device")

    with _index_lock:
        if _index_state["running"]:
            raise HTTPException(status_code=409, detail="Indexing already in progress")
        _index_state["running"] = True
        _index_state["stage"] = "pipeline"
        _index_state["scanned_pdfs"] = 0
        _index_state["scanned_pages"] = 0
        _index_state["classified_pages"] = 0
        _index_state["indexed"] = 0
        _index_state["total"] = 0
        _index_state["visual_pages"] = 0
        _index_state["errors"] = 0
        _index_state["error_message"] = None
        _index_state["scan_complete"] = False
        _index_state["classify_complete"] = False

    def _run():
        from pdfiles.pipeline import run_streaming_pipeline

        def _update_progress(pipeline_state: dict):
            for k, v in pipeline_state.items():
                if k in _index_state:
                    _index_state[k] = v

        try:
            _index_state["stage"] = "pipeline"
            run_streaming_pipeline(
                cfg=_cfg,
                embedder=_searcher.embedder,
                progress_callback=_update_progress,
                exclude_classification="TEXT_ONLY",
            )
        except Exception:
            logger.exception("Indexing pipeline failed")
            _index_state["error_message"] = traceback.format_exc()
        finally:
            _index_state["running"] = False
            _index_state["stage"] = ""

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/api/index/status")
def index_status():
    return dict(_index_state)


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    multi: bool = Query(True, description="Use multi-query RRF (better quality, 3x slower)"),
):
    if multi:
        results = _searcher.search_multi(q, top_k=top_k)
    else:
        results = _searcher.search(q, top_k=top_k)
    return [
        {
            "page_id": r.page_id,
            "pdf_path": r.pdf_path,
            "source_path": str(_resolve_pdf_path(r.pdf_path)),
            "pdf_id": r.pdf_id,
            "page_index": r.page_index,
            "total_pages": r.total_pages,
            "volume": r.volume,
            "score": r.score,
            "point_id": r.point_id,
        }
        for r in results
    ]


@app.get("/api/search/similar")
def search_similar(
    point_id: int = Query(..., description="Qdrant integer point ID"),
    top_k: int = Query(10, ge=1, le=100),
):
    from pdfiles.qdrant_store import QdrantStore

    store = QdrantStore(_cfg)
    results = store.search_similar(point_id, top_k=top_k)
    return [
        {
            "page_id": r.page_id,
            "pdf_path": r.pdf_path,
            "source_path": str(_resolve_pdf_path(r.pdf_path)),
            "pdf_id": r.pdf_id,
            "page_index": r.page_index,
            "total_pages": r.total_pages,
            "volume": r.volume,
            "score": r.score,
            "point_id": r.point_id,
        }
        for r in results
    ]


@app.get("/api/page/{point_id}/image")
def page_image(point_id: int):
    from pdfiles.qdrant_store import QdrantStore

    store = QdrantStore(_cfg)
    points = store.client.retrieve(
        collection_name=store.collection,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        raise HTTPException(status_code=404, detail=f"Page {point_id} not found")

    payload = points[0].payload
    pdf_path = _resolve_pdf_path(payload["pdf_path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")

    img = render_page(pdf_path, payload["page_index"], dpi=_cfg.render_dpi)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/clusters")
def clusters():
    if not _cfg.librarian_db.exists():
        return []
    db = LibrarianDB(_cfg.librarian_db)
    result = db.get_all_clusters()
    db.close()
    return result


@app.get("/api/clusters/{cluster_id}")
def cluster_detail(cluster_id: int):
    if not _cfg.librarian_db.exists():
        raise HTTPException(status_code=404, detail="No librarian database")
    db = LibrarianDB(_cfg.librarian_db)
    all_clusters = db.get_all_clusters()
    meta = None
    for c in all_clusters:
        if c["cluster_id"] == cluster_id:
            meta = c
            break
    if meta is None:
        db.close()
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    pages = db.get_cluster_pages(cluster_id)
    db.close()
    return {
        **meta,
        "pages": [{"page_id": pid, "distance": dist} for pid, dist in pages],
    }


@app.post("/api/export")
async def create_export():
    if not _cfg.admin_mode:
        raise HTTPException(status_code=403, detail="Admin mode is disabled")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_cfg.qdrant_url}/collections/{_cfg.collection_name}/snapshots",
            timeout=300,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to create Qdrant snapshot")
        data = resp.json()
    return {"name": data["result"]["name"]}


@app.get("/api/export/{name}")
async def download_export(name: str):
    if not _cfg.admin_mode:
        raise HTTPException(status_code=403, detail="Admin mode is disabled")
    url = f"{_cfg.qdrant_url}/collections/{_cfg.collection_name}/snapshots/{name}"

    async def _stream():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, timeout=300) as resp:
                if resp.status_code != 200:
                    raise HTTPException(status_code=502, detail="Failed to download snapshot")
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    yield chunk

    return StreamingResponse(
        _stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
