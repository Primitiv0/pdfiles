import logging
import random
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from pdfiles.config import Config
from pdfiles.opt_parser import PageRecord
from pdfiles.renderer import render_page

logger = logging.getLogger(__name__)

VISUAL_LABELS = {"Table", "Figure", "Picture", "Form", "Equation"}


class Classification(Enum):
    VISUAL = "VISUAL"
    TEXT_ONLY = "TEXT_ONLY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class PageClassification:
    page_id: str
    classification: Classification
    tier: int
    confidence: float
    text_ratio: float
    visual_area: float | None = None
    visual_labels: str | None = None


def classification_key(record: PageRecord) -> str:
    """Stable per-page key for classification persistence.

    Prefer the unique point_id. Fall back to page_id for legacy callers
    that do not populate point_id.
    """
    if record.point_id is not None:
        return str(record.point_id)
    return record.page_id


# ---------------------------------------------------------------------------
# BouncerDB -- SQLite persistence
# ---------------------------------------------------------------------------

class BouncerDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS page_classifications (
                page_id TEXT PRIMARY KEY,
                classification TEXT NOT NULL,
                tier INTEGER NOT NULL,
                confidence REAL NOT NULL,
                text_ratio REAL NOT NULL,
                visual_area REAL,
                visual_labels TEXT
            )
        """)
        self._conn.commit()

    def save_batch(self, results: list[PageClassification]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO page_classifications
               (page_id, classification, tier, confidence, text_ratio, visual_area, visual_labels)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.page_id,
                    r.classification.value,
                    r.tier,
                    r.confidence,
                    r.text_ratio,
                    r.visual_area,
                    r.visual_labels,
                )
                for r in results
            ],
        )
        self._conn.commit()

    def get_classified_ids(self) -> set[str]:
        rows = self._conn.execute("SELECT page_id FROM page_classifications").fetchall()
        return {r[0] for r in rows}

    def get_ids_by_classification(self, cls: Classification) -> set[str]:
        rows = self._conn.execute(
            "SELECT page_id FROM page_classifications WHERE classification = ?",
            (cls.value,),
        ).fetchall()
        return {r[0] for r in rows}

    def get_pages_with_label(self, label: str) -> set[str]:
        """Get page IDs whose visual_labels contain the given label."""
        rows = self._conn.execute(
            "SELECT page_id FROM page_classifications WHERE visual_labels LIKE ?",
            (f"%{label}%",),
        ).fetchall()
        return {r[0] for r in rows}

    def get_visual_labels_for_ids(self, page_ids: list[str]) -> dict[str, str | None]:
        """Get visual_labels for a batch of page IDs."""
        if not page_ids:
            return {}
        result = {}
        # Query in chunks to avoid SQLite variable limits
        chunk_size = 500
        for i in range(0, len(page_ids), chunk_size):
            chunk = page_ids[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT page_id, visual_labels FROM page_classifications WHERE page_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for pid, labels in rows:
                result[pid] = labels
        return result

    def get_stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT classification, COUNT(*) FROM page_classifications GROUP BY classification"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Tier 1 -- PyMuPDF text-ratio heuristic
# ---------------------------------------------------------------------------

def _compute_text_ratio(pdf_path: Path, page_index: int) -> float:
    """Compute the ratio of page area covered by text blocks."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        if page_area == 0:
            return 0.0

        blocks = page.get_text("blocks")
        text_area = 0.0
        for b in blocks:
            # blocks: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image
            if b[6] == 0:
                w = b[2] - b[0]
                h = b[3] - b[1]
                text_area += w * h

        return text_area / page_area
    finally:
        doc.close()


def classify_tier1(
    record: PageRecord,
    high_threshold: float,
    low_threshold: float,
) -> PageClassification:
    """Classify a single page using Tier 1 text-ratio heuristic."""
    text_ratio = _compute_text_ratio(record.pdf_path, record.page_index)

    if text_ratio >= high_threshold:
        cls = Classification.TEXT_ONLY
        confidence = min(1.0, text_ratio / high_threshold)
    elif text_ratio <= low_threshold:
        cls = Classification.VISUAL
        confidence = 1.0 - (text_ratio / low_threshold) if low_threshold > 0 else 1.0
    else:
        cls = Classification.UNCERTAIN
        # Confidence is distance from midpoint, normalized
        midpoint = (high_threshold + low_threshold) / 2
        spread = (high_threshold - low_threshold) / 2
        confidence = abs(text_ratio - midpoint) / spread if spread > 0 else 0.0

    return PageClassification(
        page_id=classification_key(record),
        classification=cls,
        tier=1,
        confidence=confidence,
        text_ratio=text_ratio,
    )


# ---------------------------------------------------------------------------
# Tier 2 -- Surya layout detection
# ---------------------------------------------------------------------------

def load_surya_predictor():
    """Load Surya LayoutPredictor (lazy, GPU-heavy)."""
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings

    foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    return LayoutPredictor(foundation)


def classify_tier2_batch(
    records: list[PageRecord],
    text_ratios: dict[str, float],
    threshold: float,
    dpi: int,
    predictor,
) -> list[PageClassification]:
    """Classify a batch of uncertain pages using Surya layout detection."""
    images = []
    valid_records = []
    for record in records:
        try:
            img = render_page(record.pdf_path, record.page_index, dpi=dpi)
            images.append(img)
            valid_records.append(record)
        except Exception:
            logger.exception("Failed to render %s page %d for Tier 2", record.pdf_path, record.page_index)

    if not images:
        return []

    layout_results = predictor(images)

    results = []
    for idx, (record, layout) in enumerate(zip(valid_records, layout_results)):
        # Compute visual element area ratio
        img = images[idx]
        total_area = img.width * img.height
        visual_area = 0.0
        labels_found = []

        for bbox in layout.bboxes:
            if bbox.label in VISUAL_LABELS:
                w = bbox.bbox[2] - bbox.bbox[0]
                h = bbox.bbox[3] - bbox.bbox[1]
                visual_area += w * h
                labels_found.append(bbox.label)

        visual_ratio = visual_area / total_area if total_area > 0 else 0.0

        if visual_ratio >= threshold:
            cls = Classification.VISUAL
        else:
            cls = Classification.TEXT_ONLY

        results.append(PageClassification(
            page_id=classification_key(record),
            classification=cls,
            tier=2,
            confidence=min(1.0, visual_ratio / threshold) if cls == Classification.VISUAL else 1.0 - visual_ratio / threshold,
            text_ratio=text_ratios.get(classification_key(record), 0.0),
            visual_area=visual_ratio,
            visual_labels=",".join(sorted(set(labels_found))) if labels_found else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Tier 1 only -- for API pipeline (CPU-fast, with progress callback)
# ---------------------------------------------------------------------------

def run_tier1_only(
    cfg: Config,
    records: list[PageRecord],
    progress_callback: callable = None,
) -> None:
    """Run Tier 1 classification with progress reporting. Skips already-classified pages.

    UNCERTAIN pages stay UNCERTAIN (Tier 2 remains CLI-only).
    """
    db = BouncerDB(cfg.bouncer_db)
    classified = db.get_classified_ids()
    remaining = [r for r in records if classification_key(r) not in classified]

    total = len(remaining)
    logger.info("Bouncer Tier 1: %d total, %d already classified, %d remaining",
                len(records), len(classified), total)

    if progress_callback:
        progress_callback(0, total)

    if not remaining:
        db.close()
        return

    batch_buffer: list[PageClassification] = []
    done = 0

    for record in remaining:
        try:
            result = classify_tier1(record, cfg.bouncer_high_threshold, cfg.bouncer_low_threshold)
            # Save UNCERTAIN pages as-is (Tier 2 is CLI-only)
            batch_buffer.append(result)

            if len(batch_buffer) >= 1000:
                db.save_batch(batch_buffer)
                batch_buffer = []
        except Exception:
            logger.exception("Tier 1 failed for %s", record.page_id)

        done += 1
        if progress_callback and done % 100 == 0:
            progress_callback(done, total)

    if batch_buffer:
        db.save_batch(batch_buffer)

    if progress_callback:
        progress_callback(total, total)

    stats = db.get_stats()
    logger.info("Tier 1 complete: %s", stats)
    db.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def classify_all(
    cfg: Config,
    records: list[PageRecord],
    tier2: bool = True,
) -> None:
    """Run full classification pipeline on all records."""
    db = BouncerDB(cfg.bouncer_db)

    # Resume: skip already-classified pages
    classified = db.get_classified_ids()
    remaining = [r for r in records if classification_key(r) not in classified]
    logger.info(
        "Bouncer: %d total, %d already classified, %d remaining",
        len(records), len(classified), len(remaining),
    )

    if not remaining:
        logger.info("All pages already classified")
        db.close()
        return

    # Tier 1
    logger.info("Running Tier 1 classification...")
    tier1_results = []
    uncertain_records = []
    text_ratios: dict[str, float] = {}
    batch_buffer: list[PageClassification] = []

    for record in tqdm(remaining, desc="Tier 1", unit="page"):
        try:
            result = classify_tier1(record, cfg.bouncer_high_threshold, cfg.bouncer_low_threshold)
            text_ratios[classification_key(record)] = result.text_ratio

            if result.classification == Classification.UNCERTAIN:
                uncertain_records.append(record)
            else:
                batch_buffer.append(result)
                tier1_results.append(result)

            # Save every 1000 definitive results
            if len(batch_buffer) >= 1000:
                db.save_batch(batch_buffer)
                batch_buffer = []
        except Exception:
            logger.exception("Tier 1 failed for %s", record.page_id)

    # Save remaining buffer
    if batch_buffer:
        db.save_batch(batch_buffer)

    definitive = len(tier1_results)
    logger.info(
        "Tier 1 complete: %d definitive (%d VISUAL, %d TEXT_ONLY), %d uncertain",
        definitive,
        sum(1 for r in tier1_results if r.classification == Classification.VISUAL),
        sum(1 for r in tier1_results if r.classification == Classification.TEXT_ONLY),
        len(uncertain_records),
    )

    # Tier 2
    if not tier2 or not uncertain_records:
        if uncertain_records and not tier2:
            logger.info("Skipping Tier 2 (%d uncertain pages left unclassified)", len(uncertain_records))
            # Save uncertain pages as-is so they show in stats
            batch_buffer = []
            for record in uncertain_records:
                batch_buffer.append(PageClassification(
                    page_id=classification_key(record),
                    classification=Classification.UNCERTAIN,
                    tier=1,
                    confidence=0.0,
                    text_ratio=text_ratios.get(classification_key(record), 0.0),
                ))
                if len(batch_buffer) >= 1000:
                    db.save_batch(batch_buffer)
                    batch_buffer = []
            if batch_buffer:
                db.save_batch(batch_buffer)
        db.close()
        return

    logger.info("Running Tier 2 on %d uncertain pages...", len(uncertain_records))
    predictor = load_surya_predictor()
    batch_size = cfg.batch_size
    tier2_buffer: list[PageClassification] = []

    for i in tqdm(range(0, len(uncertain_records), batch_size), desc="Tier 2", unit="batch"):
        batch = uncertain_records[i : i + batch_size]
        try:
            results = classify_tier2_batch(
                batch, text_ratios, cfg.bouncer_visual_area_threshold, cfg.render_dpi, predictor,
            )
            tier2_buffer.extend(results)

            if len(tier2_buffer) >= 1000:
                db.save_batch(tier2_buffer)
                tier2_buffer = []
        except Exception:
            logger.exception("Tier 2 failed for batch starting at %d", i)

    if tier2_buffer:
        db.save_batch(tier2_buffer)

    stats = db.get_stats()
    logger.info("Classification complete: %s", stats)
    db.close()


# ---------------------------------------------------------------------------
# Sampling utility
# ---------------------------------------------------------------------------

def sample_distribution(
    records: list[PageRecord],
    sample_size: int = 500,
) -> list[tuple[str, float]]:
    """Compute text_ratio for a random sample of pages, returned sorted."""
    if len(records) <= sample_size:
        sample = records
    else:
        sample = random.sample(records, sample_size)

    results = []
    for record in tqdm(sample, desc="Sampling", unit="page"):
        try:
            ratio = _compute_text_ratio(record.pdf_path, record.page_index)
            results.append((record.page_id, ratio))
        except Exception:
            logger.exception("Failed to sample %s", record.page_id)

    results.sort(key=lambda x: x[1])
    return results
