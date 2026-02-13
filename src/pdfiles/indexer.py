import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from pdfiles.config import Config
from pdfiles.embedder import Embedder
from pdfiles.opt_parser import PageRecord, load_page_records
from pdfiles.pooling import pool_batch
from pdfiles.qdrant_store import QdrantStore
from pdfiles.renderer import render_page

logger = logging.getLogger(__name__)


def run_indexing(
    cfg: Config,
    limit: int | None = None,
    filter_classification: str | None = None,
    folder: str | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    embedder: Embedder | None = None,
) -> None:
    """Run the full indexing pipeline.

    Args:
        cfg: Configuration
        limit: If set, only index this many pages (for testing)
        filter_classification: If set, only index pages with this bouncer classification (e.g. "VISUAL")
        folder: If set, only index pages from this IMAGES subfolder (e.g. "0330")
        progress_callback: If set, called with (indexed, errors, total) after each batch
        embedder: If set, reuse this embedder instead of creating a new one
    """
    # 1. Load page records (OPT manifest or directory walk)
    records = load_page_records(cfg)

    # 1a. Apply folder filter if requested
    if folder:
        records = [r for r in records if f"/{folder}/" in str(r.pdf_path)]
        logger.info("%d pages after folder filter '/%s/'", len(records), folder)

    # 1b. Apply bouncer filter if requested
    if filter_classification:
        from pdfiles.bouncer import BouncerDB, Classification, classification_key
        logger.info("Filtering by bouncer classification: %s", filter_classification)
        db = BouncerDB(cfg.bouncer_db)
        allowed_ids = db.get_ids_by_classification(Classification(filter_classification))
        db.close()
        before_count = len(records)
        records = [r for r in records if classification_key(r) in allowed_ids]
        logger.info("%d pages after bouncer filter", len(records))
        if len(records) == 0 and before_count > 0 and len(allowed_ids) > 0:
            logger.error(
                "Zero records matched bouncer filter! "
                "Bouncer DB has %d %s entries but none matched %d page records. "
                "This likely means the index IDs don't match the bouncer DB IDs. "
                "Re-index with correct IDs first.",
                len(allowed_ids), filter_classification, before_count,
            )
            return

    # 2. Set up Qdrant
    store = QdrantStore(cfg)
    store.ensure_collection()

    # 3. Check already-indexed (resume support)
    logger.info("Checking already-indexed pages...")
    indexed_ids = store.get_indexed_ids()
    logger.info("Found %d already-indexed pages", len(indexed_ids))

    # 4. Filter out already-indexed records
    remaining = [r for r in records if r.point_id not in indexed_ids]
    if limit is not None:
        remaining = remaining[:limit]
    logger.info("%d pages remaining to index", len(remaining))

    if not remaining:
        logger.info("Nothing to index, all pages already indexed")
        return

    # 5. Group by PDF path for efficient rendering
    pdf_groups: dict[Path, list[PageRecord]] = defaultdict(list)
    for r in remaining:
        pdf_groups[r.pdf_path].append(r)

    # 6. Load embedder (reuse if provided)
    if embedder is None:
        embedder = Embedder(cfg)

    # 7. Process in batches
    batch_images = []
    batch_records: list[PageRecord] = []
    errors = 0
    indexed = 0

    progress = tqdm(total=len(remaining), desc="Indexing", unit="page")

    for pdf_path, page_records in pdf_groups.items():
        for record in page_records:
            try:
                img = render_page(record.pdf_path, record.page_index, dpi=cfg.render_dpi)
                batch_images.append(img)
                batch_records.append(record)
            except Exception:
                logger.exception("Failed to render %s page %d", record.pdf_path, record.page_index)
                errors += 1
                progress.update(1)
                continue

            # Process batch when full
            if len(batch_images) >= cfg.batch_size:
                try:
                    _process_batch(embedder, store, cfg, batch_images, batch_records)
                    indexed += len(batch_images)
                except Exception:
                    logger.exception("Failed to process batch")
                    errors += len(batch_images)
                progress.update(len(batch_images))
                if progress_callback:
                    progress_callback(indexed, errors, len(remaining))
                batch_images = []
                batch_records = []

    # Process remaining partial batch
    if batch_images:
        try:
            _process_batch(embedder, store, cfg, batch_images, batch_records)
            indexed += len(batch_images)
        except Exception:
            logger.exception("Failed to process final batch")
            errors += len(batch_images)
        progress.update(len(batch_images))
        if progress_callback:
            progress_callback(indexed, errors, len(remaining))

    progress.close()
    logger.info("Indexing complete: %d indexed, %d errors", indexed, errors)


def _process_batch(
    embedder: Embedder,
    store: QdrantStore,
    cfg: Config,
    images: list,
    records: list[PageRecord],
) -> None:
    """Embed, pool, and upsert a batch of images."""
    # Embed
    embeddings = embedder.embed_images(images)

    # Pool
    if cfg.use_pooling:
        embeddings = pool_batch(embeddings, cfg)

    # Convert to nested lists for Qdrant
    vectors = [emb.tolist() for emb in embeddings]

    # Build payloads (store paths relative to data_root for portability)
    point_ids = [r.point_id for r in records]
    payloads = []
    for r in records:
        try:
            rel_path = str(r.pdf_path.relative_to(cfg.data_root))
        except ValueError:
            rel_path = str(r.pdf_path)
        payloads.append({
            "page_id": r.page_id,
            "pdf_path": rel_path,
            "page_index": r.page_index,
            "pdf_id": r.pdf_id,
            "volume": r.volume,
            "total_pages": r.total_pages,
        })

    # Upsert
    store.upsert_batch(point_ids, vectors, payloads)
