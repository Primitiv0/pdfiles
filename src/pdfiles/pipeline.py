"""Streaming 3-thread pipeline: Scanner -> Classifier -> Indexer.

Pages become searchable within seconds of starting. Killing at any point
loses at most ~4 pages of work (one GPU batch).
"""

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz

from pdfiles.bouncer import (
    BouncerDB,
    Classification,
    PageClassification,
    classify_tier1,
    classification_key,
)
from pdfiles.config import Config
from pdfiles.embedder import Embedder
from pdfiles.indexer import _process_batch
from pdfiles.manifest import ManifestDB
from pdfiles.opt_parser import (
    PageRecord,
    discover_opt_files,
    infer_opt_volume_root,
    iter_pdfs_excluding_roots,
    parse_opt_pdfs,
)
from pdfiles.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Queue sentinel
_SENTINEL = None


def run_streaming_pipeline(
    cfg: Config,
    embedder: Embedder,
    progress_callback: Callable[[dict], None] | None = None,
    exclude_classification: str = "TEXT_ONLY",
) -> None:
    """Run the streaming 3-thread pipeline.

    Args:
        cfg: Configuration
        embedder: Shared embedder (already loaded on GPU)
        progress_callback: Called with state dict for progress updates
        exclude_classification: Skip pages with this bouncer classification
    """
    state = {
        "scanned_pdfs": 0,
        "scanned_pages": 0,
        "classified_pages": 0,
        "indexed": 0,
        "total": 0,
        "visual_pages": 0,
        "errors": 0,
        "scan_complete": False,
        "classify_complete": False,
    }

    def _notify():
        if progress_callback:
            progress_callback(dict(state))

    # Shared queues
    scan_queue: queue.Queue[list[PageRecord] | None] = queue.Queue()
    classify_queue: queue.Queue[PageRecord | None] = queue.Queue(maxsize=100)

    # Shared error tracking
    scanner_error: list[Exception | None] = [None]
    classifier_error: list[Exception | None] = [None]

    # ---------------------------------------------------------------
    # Thread 1: Scanner
    # ---------------------------------------------------------------
    def _scanner():
        try:
            manifest = ManifestDB(cfg.manifest_db)
            known_paths = {str(Path(p).resolve()) for p in manifest.get_known_paths()}
            state["scanned_pdfs"] = manifest.count_pdfs()
            state["scanned_pages"] = manifest.count_pages()

            # Replay known PDFs first so downstream work can start immediately.
            if known_paths:
                logger.info("Replaying %d previously scanned PDFs", len(known_paths))
                for batch in manifest.iter_page_record_batches(batch_size=100):
                    scan_queue.put(batch)
                _notify()

            logger.info("Scanning for new PDFs under %s", cfg.data_root)

            def _enqueue_new_pdf(pdf_path: Path, known_page_count: int | None = None) -> None:
                pdf_path = pdf_path.resolve()
                pdf_str = str(pdf_path)
                if pdf_str in known_paths:
                    return
                if pdf_path.suffix.lower() != ".pdf":
                    return
                if not pdf_path.exists():
                    logger.warning("Skipping missing PDF: %s", pdf_path)
                    state["errors"] += 1
                    return

                page_count = known_page_count
                if page_count is None:
                    try:
                        doc = fitz.open(pdf_path)
                        page_count = len(doc)
                        doc.close()
                    except Exception:
                        logger.warning("Skipping unreadable PDF: %s", pdf_path)
                        state["errors"] += 1
                        return
                elif page_count <= 0:
                    logger.warning("Skipping OPT entry with invalid page count: %s", pdf_path)
                    state["errors"] += 1
                    return

                first_page_id = manifest.get_next_id()
                manifest.insert_pdf(pdf_str, page_count, first_page_id)
                known_paths.add(pdf_str)

                batch = []
                for page_idx in range(page_count):
                    batch.append(PageRecord(
                        page_id=pdf_path.stem,  # display label
                        volume="LOCAL",
                        pdf_path=pdf_path,
                        page_index=page_idx,
                        pdf_id=pdf_path.stem,
                        total_pages=page_count,
                        point_id=first_page_id + page_idx,
                    ))
                scan_queue.put(batch)

                state["scanned_pdfs"] += 1
                state["scanned_pages"] += page_count
                if state["scanned_pdfs"] % 25 == 0:
                    _notify()

            # Phase 1: OPT manifests (preferred) and mark covered roots.
            covered_roots: list[Path] = []
            opt_files = discover_opt_files(cfg.data_root)
            if opt_files:
                logger.info("Found %d OPT manifest(s)", len(opt_files))

            for opt_path in opt_files:
                volume_root = infer_opt_volume_root(opt_path)
                covered_roots.append(volume_root)
                images_root = volume_root / "IMAGES"
                try:
                    opt_pdfs = parse_opt_pdfs(opt_path, images_root)
                except Exception:
                    logger.exception("Failed parsing OPT manifest: %s", opt_path)
                    state["errors"] += 1
                    continue

                for pdf_path, page_count in opt_pdfs:
                    _enqueue_new_pdf(pdf_path, known_page_count=page_count)

            # Phase 2: Walk remaining uncovered roots for extra PDFs.
            for pdf_path in iter_pdfs_excluding_roots(cfg.data_root, covered_roots):
                _enqueue_new_pdf(pdf_path)

            manifest.mark_scan_complete()
            state["scan_complete"] = True
            _notify()
            logger.info(
                "Scan complete: %d PDFs, %d pages",
                state["scanned_pdfs"],
                state["scanned_pages"],
            )
            manifest.close()

        except Exception as e:
            logger.exception("Scanner failed")
            scanner_error[0] = e
        finally:
            scan_queue.put(_SENTINEL)

    # ---------------------------------------------------------------
    # Thread 2: Classifier
    # ---------------------------------------------------------------
    def _classifier():
        try:
            bouncer_db = BouncerDB(cfg.bouncer_db)
            classified_ids = bouncer_db.get_classified_ids()

            # Pre-load which IDs are TEXT_ONLY (to exclude)
            excluded_cls = Classification(exclude_classification)
            text_only_ids = bouncer_db.get_ids_by_classification(excluded_cls)

            batch_buffer: list[PageClassification] = []
            indexable_count = 0

            while True:
                try:
                    batch = scan_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                if batch is _SENTINEL:
                    break

                for record in batch:
                    record_cls_id = classification_key(record)
                    if record_cls_id in classified_ids:
                        # Already classified -- forward unless TEXT_ONLY
                        if record_cls_id not in text_only_ids:
                            classify_queue.put(record)
                        state["classified_pages"] += 1
                        continue

                    try:
                        result = classify_tier1(
                            record,
                            cfg.bouncer_high_threshold,
                            cfg.bouncer_low_threshold,
                        )
                        batch_buffer.append(result)
                        classified_ids.add(result.page_id)

                        if result.classification == excluded_cls:
                            text_only_ids.add(result.page_id)
                        else:
                            classify_queue.put(record)
                            indexable_count += 1

                        if len(batch_buffer) >= 1000:
                            bouncer_db.save_batch(batch_buffer)
                            batch_buffer = []

                    except Exception:
                        logger.exception("Tier 1 failed for %s page %d", record.pdf_path, record.page_index)
                        state["errors"] += 1

                    state["classified_pages"] += 1

                indexable = len(classified_ids - text_only_ids)
                state["visual_pages"] = indexable
                state["total"] = indexable
                _notify()

            # Flush remaining
            if batch_buffer:
                bouncer_db.save_batch(batch_buffer)

            indexable = len(classified_ids - text_only_ids)
            state["visual_pages"] = indexable
            state["total"] = indexable
            state["classify_complete"] = True
            _notify()

            stats = bouncer_db.get_stats()
            logger.info("Classification complete: %s (newly indexable: %d)", stats, indexable_count)
            bouncer_db.close()

        except Exception as e:
            logger.exception("Classifier failed")
            classifier_error[0] = e
        finally:
            classify_queue.put(_SENTINEL)

    # ---------------------------------------------------------------
    # Thread 3 (main thread): Indexer
    # ---------------------------------------------------------------
    store = QdrantStore(cfg)
    store.ensure_collection()

    logger.info("Loading indexed IDs from Qdrant for resume...")
    indexed_ids = store.get_indexed_ids()
    state["indexed"] = len(indexed_ids)
    logger.info("Found %d already-indexed pages", len(indexed_ids))
    _notify()

    # Start scanner and classifier threads
    scanner_thread = threading.Thread(target=_scanner, daemon=True, name="scanner")
    classifier_thread = threading.Thread(target=_classifier, daemon=True, name="classifier")
    scanner_thread.start()
    classifier_thread.start()

    # Concurrent rendering + GPU batching
    # Render threads push completed (record, image) to render_queue.
    # Main thread drains in completion order -- no head-of-line blocking.
    from pdfiles.renderer import render_page

    render_workers = cfg.num_render_workers  # default 11
    render_queue: queue.Queue[tuple[PageRecord, object] | None] = queue.Queue(maxsize=render_workers * 2)
    render_inflight = threading.Semaphore(render_workers * 2)  # cap outstanding renders
    inflight_lock = threading.Lock()
    inflight_count = 0
    render_pool = ThreadPoolExecutor(max_workers=render_workers, thread_name_prefix="render")

    def _render_and_enqueue(record: PageRecord):
        """Render one page then push result to render_queue (runs in thread pool)."""
        nonlocal inflight_count
        try:
            img = render_page(record.pdf_path, record.page_index, dpi=cfg.render_dpi)
            render_queue.put((record, img))
        except Exception:
            logger.exception("Failed to render %s page %d", record.pdf_path, record.page_index)
            render_queue.put(None)  # signal error, still unblock the consumer
        finally:
            with inflight_lock:
                inflight_count -= 1
            render_inflight.release()

    # Feeder thread: reads classify_queue -> submits to render pool
    feeder_done = threading.Event()

    def _render_feeder():
        nonlocal inflight_count
        try:
            while True:
                try:
                    record = classify_queue.get(timeout=2.0)
                except queue.Empty:
                    if classifier_error[0] is not None:
                        break
                    continue

                if record is _SENTINEL:
                    break

                if record.point_id in indexed_ids:
                    continue

                render_inflight.acquire()  # backpressure if too many in flight
                with inflight_lock:
                    inflight_count += 1
                render_pool.submit(_render_and_enqueue, record)
        finally:
            feeder_done.set()

    feeder_thread = threading.Thread(target=_render_feeder, daemon=True, name="render-feeder")
    feeder_thread.start()

    # Main loop: drain render_queue into GPU batches
    batch_images = []
    batch_records: list[PageRecord] = []

    while True:
        try:
            item = render_queue.get(timeout=2.0)
        except queue.Empty:
            with inflight_lock:
                no_renders_left = inflight_count == 0
            if feeder_done.is_set() and no_renders_left and render_queue.empty():
                break
            continue

        if item is None:
            # Render error -- already logged in _render_and_enqueue
            state["errors"] += 1
            continue

        record, img = item
        batch_images.append(img)
        batch_records.append(record)

        if len(batch_images) >= cfg.batch_size:
            try:
                _process_batch(embedder, store, cfg, batch_images, batch_records)
                state["indexed"] += len(batch_images)
                for r in batch_records:
                    indexed_ids.add(r.point_id)
                _notify()
            except Exception:
                logger.exception("Failed to process batch")
                state["errors"] += len(batch_images)
            batch_images = []
            batch_records = []

    # Flush remaining partial batch
    if batch_images:
        try:
            _process_batch(embedder, store, cfg, batch_images, batch_records)
            state["indexed"] += len(batch_images)
            for r in batch_records:
                indexed_ids.add(r.point_id)
            _notify()
        except Exception:
            logger.exception("Failed to process final batch")
            state["errors"] += len(batch_images)

    feeder_thread.join(timeout=5.0)
    render_pool.shutdown(wait=False)

    # Wait for threads
    scanner_thread.join(timeout=5.0)
    classifier_thread.join(timeout=5.0)

    # Propagate errors
    if scanner_error[0]:
        logger.error("Scanner had error: %s", scanner_error[0])
    if classifier_error[0]:
        logger.error("Classifier had error: %s", classifier_error[0])

    logger.info(
        "Pipeline complete: %d indexed, %d visual, %d errors",
        state["indexed"],
        state["visual_pages"],
        state["errors"],
    )
