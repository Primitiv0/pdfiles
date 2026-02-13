#!/usr/bin/env python3
"""Search Lab -- diagnostic Gradio app for visual search quality analysis.

Dev-only tool for comparing queries side-by-side, inspecting score distributions,
detecting result overlap, and browsing the full collection.

Usage:
    python tools/search_lab.py [--cpu]
"""

import logging
import os
import random
import sys
from pathlib import Path

import gradio as gr
import numpy as np

# Load .env from project root (so DATA_ROOT is available)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from pdfiles.config import Config
from pdfiles.qdrant_store import QdrantStore, SearchResult
from pdfiles.renderer import render_page
from pdfiles.searcher import Searcher, QUERY_VARIANTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cpu" if "--cpu" in sys.argv else "cuda:0"
cfg = Config(device=device)
store = QdrantStore(cfg)

# Lazy-load embedder only when search is needed (saves VRAM for browse-only use)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from pdfiles.embedder import Embedder
        _embedder = Embedder(cfg)
    return _embedder


# Old expansion template kept here for A/B testing
EXPANSION_TEMPLATE = "Find a scanned document page showing: {query}"

# Lazy-load searcher for multi-query RRF (reuses embedder)
_searcher = None


def _get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher(cfg)
    return _searcher

# Cache of all point IDs for the browse tab
_all_point_ids: list[int] | None = None


def _get_all_point_ids() -> list[int]:
    """Get all point IDs, cached after first fetch."""
    global _all_point_ids
    if _all_point_ids is None:
        logger.info("Fetching all point IDs for browse...")
        _all_point_ids = sorted(store.get_indexed_ids())
        logger.info("Cached %d point IDs", len(_all_point_ids))
    return _all_point_ids


def _search(query: str, use_expansion: bool, top_k: int) -> list[SearchResult]:
    """Run a search with optional expansion template."""
    if not query.strip():
        return []
    effective = EXPANSION_TEMPLATE.format(query=query) if use_expansion else query
    embedder = _get_embedder()
    query_emb = embedder.embed_query(effective)
    return store.search(query_emb.tolist(), top_k=top_k)


def _render_point(point_id: int) -> tuple | None:
    """Render a single point by ID, returning (image, caption) or None."""
    try:
        points = store.client.retrieve(
            collection_name=store.collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = points[0].payload
        img = render_page(
            cfg.resolve_pdf_path(payload["pdf_path"]),
            payload["page_index"],
            dpi=cfg.render_dpi,
        )
        caption = (
            f"{payload['page_id']} | "
            f"Page {payload['page_index'] + 1}/{payload['total_pages']} | "
            f"PDF {payload['pdf_id']}"
        )
        return (img, caption)
    except Exception as e:
        logger.error("Render failed for point %d: %s", point_id, e)
        return None


def _render_results(
    results: list[SearchResult], overlap_ids: set[str] | None = None
) -> list[tuple]:
    """Render search results as gallery items. Overlap IDs get a red border marker."""
    items = []
    for r in results:
        try:
            img = render_page(cfg.resolve_pdf_path(r.pdf_path), r.page_index, dpi=cfg.render_dpi)
            marker = " [OVERLAP]" if overlap_ids and r.page_id in overlap_ids else ""
            caption = (
                f"{'>> ' if marker else ''}"
                f"Score: {r.score:.4f} | {r.page_id} | "
                f"Page {r.page_index + 1}/{r.total_pages}{marker}"
            )
            items.append((img, caption))
        except Exception as e:
            logger.error("Render failed for %s: %s", r.page_id, e)
    return items


def _score_stats(results: list[SearchResult]) -> str:
    """Format score distribution statistics."""
    if not results:
        return "No results"
    scores = [r.score for r in results]
    return (
        f"**Results:** {len(scores)}  \n"
        f"**Max:** {max(scores):.4f}  \n"
        f"**Min:** {min(scores):.4f}  \n"
        f"**Spread:** {max(scores) - min(scores):.4f}  \n"
        f"**Mean:** {np.mean(scores):.4f}  \n"
        f"**Std:** {np.std(scores):.4f}"
    )


def _overlap_report(results_a: list[SearchResult], results_b: list[SearchResult]) -> str:
    """Generate overlap report between two result sets."""
    if not results_a or not results_b:
        return "Run both queries to see overlap."
    ids_a = {r.page_id for r in results_a}
    ids_b = {r.page_id for r in results_b}
    shared = ids_a & ids_b
    total = len(ids_a | ids_b)
    if not shared:
        return f"**No overlap** -- 0 of {total} unique results shared."
    pct = len(shared) / min(len(ids_a), len(ids_b)) * 100
    shared_list = ", ".join(sorted(shared))
    return (
        f"**{len(shared)} of {total} unique results appear in both queries** ({pct:.0f}% of smaller set)  \n"
        f"Shared IDs: {shared_list}"
    )


def collection_stats() -> str:
    """Get collection health stats."""
    try:
        info = store.client.get_collection(store.collection)
        status = info.status.value if hasattr(info.status, "value") else str(info.status)
        return (
            f"**Collection:** {store.collection}  \n"
            f"**Points:** {info.points_count:,}  \n"
            f"**Status:** {status}  \n"
            f"**Vectors:** {cfg.vector_dim}d x {cfg.pooled_vectors} multi-vectors  \n"
            f"**Qdrant:** {cfg.qdrant_url}"
        )
    except Exception as e:
        return f"**Error connecting:** {e}"


def run_comparison(query_a: str, query_b: str, use_expansion: bool, top_k: int):
    """Run both queries and return all outputs."""
    results_a = _search(query_a, use_expansion, int(top_k))
    results_b = _search(query_b, use_expansion, int(top_k))

    ids_a = {r.page_id for r in results_a}
    ids_b = {r.page_id for r in results_b}
    overlap_ids = ids_a & ids_b

    gallery_a = _render_results(results_a, overlap_ids)
    gallery_b = _render_results(results_b, overlap_ids)
    stats_a = _score_stats(results_a)
    stats_b = _score_stats(results_b)
    overlap = _overlap_report(results_a, results_b)

    if use_expansion:
        expansion_note = (
            f"**Expansion ON** -- queries wrapped as:  \n"
            f'`"{EXPANSION_TEMPLATE.format(query="<query>")}"`'
        )
    else:
        expansion_note = "**Expansion OFF** -- raw queries sent to ColQwen2.5"

    return gallery_a, stats_a, gallery_b, stats_b, overlap, expansion_note


# ---------------------------------------------------------------------------
# Browse Collection
# ---------------------------------------------------------------------------

# Low DPI for thumbnails -- fast rendering, good enough for eyeballing
BROWSE_DPI = 72


def _render_points(point_ids: list[int]) -> tuple[list[tuple], int]:
    """Batch-retrieve and render a list of point IDs. Returns (items, error_count)."""
    # Qdrant retrieve has no built-in limit, but do it in chunks to avoid huge payloads
    CHUNK = 500
    all_points = []
    for i in range(0, len(point_ids), CHUNK):
        chunk_ids = point_ids[i : i + CHUNK]
        pts = store.client.retrieve(
            collection_name=store.collection,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(pts)

    items = []
    errors = 0
    for p in all_points:
        try:
            img = render_page(
                cfg.resolve_pdf_path(p.payload["pdf_path"]),
                p.payload["page_index"],
                dpi=BROWSE_DPI,
            )
            caption = (
                f"{p.payload['page_id']} | "
                f"Page {p.payload['page_index'] + 1}/{p.payload['total_pages']} | "
                f"PDF {p.payload['pdf_id']}"
            )
            items.append((img, caption))
        except Exception as e:
            errors += 1
            logger.error("Render failed for %s: %s", p.payload.get("page_id", "?"), e)
    return items, errors


def browse_page(page_num: int, count: int) -> tuple[list[tuple], str]:
    """Render a page of collection results."""
    count = int(count)
    all_ids = _get_all_point_ids()
    total_pages = max(1, (len(all_ids) + count - 1) // count)
    page_num = max(0, min(int(page_num), total_pages - 1))

    start = page_num * count
    end = min(start + count, len(all_ids))
    page_ids = all_ids[start:end]

    items, errors = _render_points(page_ids)

    status = (
        f"**Page {page_num + 1} of {total_pages:,}** | "
        f"Showing points {start + 1:,}-{end:,} of {len(all_ids):,} total | "
        f"{BROWSE_DPI} DPI thumbnails"
    )
    if errors:
        status += f" | **{errors} render errors**"

    return items, status


def browse_random(count: int) -> tuple[list[tuple], str]:
    """Render a random sample from across the collection."""
    count = int(count)
    all_ids = _get_all_point_ids()
    sample_ids = random.sample(all_ids, min(count, len(all_ids)))

    items, errors = _render_points(sample_ids)

    status = (
        f"**Random sample: {len(items)} rendered** of {len(all_ids):,} total | "
        f"{BROWSE_DPI} DPI thumbnails"
    )
    if errors:
        status += f" | **{errors} render errors**"

    return items, status


# ---------------------------------------------------------------------------
# Classify Sample (Tier 1 on random indexed pages)
# ---------------------------------------------------------------------------

def classify_sample(count: int) -> tuple[list[tuple], list[tuple], list[tuple], str]:
    """Pull random pages from Qdrant and classify them with Tier 1 (text_ratio).

    Returns (visual_gallery, uncertain_gallery, text_gallery, stats_md).
    No GPU needed -- uses PyMuPDF text blocks only.
    """
    from pdfiles.bouncer import classify_tier1
    from pdfiles.opt_parser import PageRecord

    count = int(count)
    all_ids = _get_all_point_ids()
    sample_ids = random.sample(all_ids, min(count, len(all_ids)))

    # Retrieve payloads from Qdrant
    CHUNK = 500
    all_points = []
    for i in range(0, len(sample_ids), CHUNK):
        chunk = sample_ids[i : i + CHUNK]
        pts = store.client.retrieve(
            collection_name=store.collection,
            ids=chunk,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(pts)

    buckets: dict[str, list[tuple]] = {"VISUAL": [], "UNCERTAIN": [], "TEXT_ONLY": []}
    errors = 0

    for p in all_points:
        payload = p.payload
        try:
            pdf_path = cfg.resolve_pdf_path(payload["pdf_path"])
            record = PageRecord(
                page_id=payload["page_id"],
                volume=payload.get("volume", "LOCAL"),
                pdf_path=pdf_path,
                page_index=payload["page_index"],
                pdf_id=payload["pdf_id"],
                total_pages=payload["total_pages"],
            )
            result = classify_tier1(
                record,
                cfg.bouncer_high_threshold,
                cfg.bouncer_low_threshold,
            )
            img = render_page(pdf_path, payload["page_index"], dpi=BROWSE_DPI)
            caption = (
                f"{payload['page_id']} | "
                f"text_ratio={result.text_ratio:.3f} | "
                f"{result.classification.value} ({result.confidence:.2f})"
            )
            buckets[result.classification.value].append((img, caption))
        except Exception as e:
            errors += 1
            logger.error("Classify failed for %s: %s", payload.get("page_id", "?"), e)

    total = sum(len(v) for v in buckets.values())
    stats = (
        f"**Sampled:** {total} pages  \n"
        f"**VISUAL:** {len(buckets['VISUAL'])} ({len(buckets['VISUAL']) / total * 100:.0f}%)  \n"
        f"**UNCERTAIN:** {len(buckets['UNCERTAIN'])} ({len(buckets['UNCERTAIN']) / total * 100:.0f}%)  \n"
        f"**TEXT_ONLY:** {len(buckets['TEXT_ONLY'])} ({len(buckets['TEXT_ONLY']) / total * 100:.0f}%)  \n"
        f"**Thresholds:** high={cfg.bouncer_high_threshold}, low={cfg.bouncer_low_threshold}"
    ) if total > 0 else "No pages classified"
    if errors:
        stats += f"  \n**Errors:** {errors}"

    return buckets["VISUAL"], buckets["UNCERTAIN"], buckets["TEXT_ONLY"], stats


# ---------------------------------------------------------------------------
# Multi-query RRF comparison
# ---------------------------------------------------------------------------


def run_multi_comparison(query: str, top_k: int):
    """Compare single-query (with expansion) vs multi-query RRF side by side."""
    if not query.strip():
        return [], "", [], "", ""

    searcher = _get_searcher()

    # Single query with expansion
    results_single = searcher.search(query, top_k=int(top_k))
    # Multi-query RRF
    results_multi = searcher.search_multi(query, top_k=int(top_k))

    ids_single = {r.page_id for r in results_single}
    ids_multi = {r.page_id for r in results_multi}
    overlap_ids = ids_single & ids_multi

    gallery_single = _render_results(results_single, overlap_ids)
    gallery_multi = _render_results(results_multi, overlap_ids)
    stats_single = _score_stats(results_single)
    stats_multi = _score_stats(results_multi)
    overlap = _overlap_report(results_single, results_multi)

    return gallery_single, stats_single, gallery_multi, stats_multi, overlap


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Search Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Search Lab")
    gr.Markdown("Diagnostic tool for search quality analysis and collection browsing.")

    stats_display = gr.Markdown(value=collection_stats())

    with gr.Tab("Compare Queries"):
        with gr.Row():
            query_a = gr.Textbox(label="Query A", placeholder="e.g., surveillance photograph", scale=3)
            query_b = gr.Textbox(label="Query B", placeholder="e.g., financial spreadsheet", scale=3)

        with gr.Row():
            use_expansion = gr.Checkbox(label="Use old expansion template (A/B test)", value=False)
            top_k = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="Top K")
            run_btn = gr.Button("Compare", variant="primary")

        expansion_note = gr.Markdown()
        overlap_report = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Query A Results")
                stats_a = gr.Markdown()
                gallery_a = gr.Gallery(label="Query A", columns=2, height="auto", object_fit="contain")

            with gr.Column():
                gr.Markdown("### Query B Results")
                stats_b = gr.Markdown()
                gallery_b = gr.Gallery(label="Query B", columns=2, height="auto", object_fit="contain")

        run_btn.click(
            fn=run_comparison,
            inputs=[query_a, query_b, use_expansion, top_k],
            outputs=[gallery_a, stats_a, gallery_b, stats_b, overlap_report, expansion_note],
        )

    with gr.Tab("Multi-Query RRF"):
        gr.Markdown(
            "Compare **single-query** (with expansion) vs **multi-query RRF** (3 variants merged). "
            "RRF runs the query raw, with photo-context, and with document-context, then merges by reciprocal rank."
        )

        with gr.Row():
            rrf_query = gr.Textbox(label="Query", placeholder="e.g., bikini, woman, graph", scale=3)
            rrf_top_k = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="Top K")
            rrf_btn = gr.Button("Compare Single vs RRF", variant="primary")

        rrf_overlap = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Single Query (expanded)")
                rrf_stats_single = gr.Markdown()
                rrf_gallery_single = gr.Gallery(label="Single", columns=2, height="auto", object_fit="contain")

            with gr.Column():
                gr.Markdown("### Multi-Query RRF")
                rrf_stats_multi = gr.Markdown()
                rrf_gallery_multi = gr.Gallery(label="RRF", columns=2, height="auto", object_fit="contain")

        rrf_btn.click(
            fn=run_multi_comparison,
            inputs=[rrf_query, rrf_top_k],
            outputs=[rrf_gallery_single, rrf_stats_single, rrf_gallery_multi, rrf_stats_multi, rrf_overlap],
        )

    with gr.Tab("Browse Collection"):
        gr.Markdown(
            "Browse all indexed pages at 72 DPI thumbnails. "
            "**Random Sample** pulls from across the entire collection. "
            "**Paginate** walks through sequentially."
        )

        browse_status = gr.Markdown()

        with gr.Row():
            browse_count = gr.Slider(
                minimum=50, maximum=2000, value=200, step=50,
                label="Pages to show",
            )
            random_btn = gr.Button("Random Sample", variant="primary")

        with gr.Row():
            page_num_input = gr.Number(label="Page #", value=0, precision=0, minimum=0)
            goto_btn = gr.Button("Go to Page")
            prev_btn = gr.Button("< Prev")
            next_btn = gr.Button("Next >")

        browse_gallery = gr.Gallery(
            label="Collection Pages",
            columns=8,
            height="auto",
            object_fit="contain",
        )

        random_btn.click(
            fn=browse_random,
            inputs=[browse_count],
            outputs=[browse_gallery, browse_status],
        )
        goto_btn.click(
            fn=browse_page,
            inputs=[page_num_input, browse_count],
            outputs=[browse_gallery, browse_status],
        )
        prev_btn.click(
            fn=lambda p, c: browse_page(max(0, int(p) - 1), c),
            inputs=[page_num_input, browse_count],
            outputs=[browse_gallery, browse_status],
        )
        next_btn.click(
            fn=lambda p, c: browse_page(int(p) + 1, c),
            inputs=[page_num_input, browse_count],
            outputs=[browse_gallery, browse_status],
        )

    with gr.Tab("Classify Sample"):
        gr.Markdown(
            "Run Tier 1 classification (PyMuPDF text_ratio) on random indexed pages. "
            "**No GPU needed** -- eyeball whether TEXT/VISUAL/UNCERTAIN buckets look right."
        )

        classify_status = gr.Markdown()

        with gr.Row():
            classify_count = gr.Slider(
                minimum=20, maximum=200, value=50, step=10,
                label="Sample size",
            )
            classify_btn = gr.Button("Classify Random Sample", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### VISUAL")
                classify_visual = gr.Gallery(label="Visual", columns=4, height="auto", object_fit="contain")
            with gr.Column():
                gr.Markdown("### UNCERTAIN")
                classify_uncertain = gr.Gallery(label="Uncertain", columns=4, height="auto", object_fit="contain")
            with gr.Column():
                gr.Markdown("### TEXT_ONLY")
                classify_text = gr.Gallery(label="Text Only", columns=4, height="auto", object_fit="contain")

        classify_btn.click(
            fn=classify_sample,
            inputs=[classify_count],
            outputs=[classify_visual, classify_uncertain, classify_text, classify_status],
        )

    refresh_btn = gr.Button("Refresh Collection Stats", size="sm")
    refresh_btn.click(fn=collection_stats, outputs=stats_display)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
