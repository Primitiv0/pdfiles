import sqlite3
from pathlib import Path

import pytest

from pdfiles.bouncer import (
    BouncerDB,
    Classification,
    PageClassification,
    _compute_text_ratio,
    classify_tier1,
    classification_key,
    sample_distribution,
)
from pdfiles.opt_parser import PageRecord

TEST_PDF = Path(__file__).parent / "fixtures" / "test.pdf"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_bouncer.db"


@pytest.fixture
def records():
    """Build PageRecord list from the test fixture PDF."""
    import fitz
    doc = fitz.open(TEST_PDF)
    page_count = len(doc)
    doc.close()
    return [
        PageRecord(
            page_id=str(i),
            volume="TEST",
            pdf_path=TEST_PDF,
            page_index=i,
            pdf_id="0",
            total_pages=page_count,
        )
        for i in range(page_count)
    ]


# ---------------------------------------------------------------------------
# BouncerDB tests
# ---------------------------------------------------------------------------


class TestBouncerDB:
    def test_create_and_save(self, db_path):
        db = BouncerDB(db_path)
        results = [
            PageClassification("1", Classification.VISUAL, 1, 0.95, 0.02),
            PageClassification("2", Classification.TEXT_ONLY, 1, 0.85, 0.45),
        ]
        db.save_batch(results)

        ids = db.get_classified_ids()
        assert ids == {"1", "2"}
        db.close()

    def test_get_ids_by_classification(self, db_path):
        db = BouncerDB(db_path)
        results = [
            PageClassification("1", Classification.VISUAL, 1, 0.95, 0.02),
            PageClassification("2", Classification.TEXT_ONLY, 1, 0.85, 0.45),
            PageClassification("3", Classification.VISUAL, 1, 0.90, 0.01),
        ]
        db.save_batch(results)

        visual = db.get_ids_by_classification(Classification.VISUAL)
        assert visual == {"1", "3"}

        text = db.get_ids_by_classification(Classification.TEXT_ONLY)
        assert text == {"2"}
        db.close()

    def test_get_stats(self, db_path):
        db = BouncerDB(db_path)
        results = [
            PageClassification("1", Classification.VISUAL, 1, 0.95, 0.02),
            PageClassification("2", Classification.TEXT_ONLY, 1, 0.85, 0.45),
            PageClassification("3", Classification.VISUAL, 1, 0.90, 0.01),
            PageClassification("4", Classification.UNCERTAIN, 1, 0.50, 0.15),
        ]
        db.save_batch(results)

        stats = db.get_stats()
        assert stats == {"VISUAL": 2, "TEXT_ONLY": 1, "UNCERTAIN": 1}
        db.close()

    def test_upsert_replaces(self, db_path):
        db = BouncerDB(db_path)
        db.save_batch([PageClassification("1", Classification.UNCERTAIN, 1, 0.5, 0.15)])
        db.save_batch([PageClassification("1", Classification.VISUAL, 2, 0.95, 0.15, 0.08, "Table")])

        stats = db.get_stats()
        assert stats == {"VISUAL": 1}

        visual = db.get_ids_by_classification(Classification.VISUAL)
        assert "1" in visual
        db.close()

    def test_wal_journal_mode(self, db_path):
        db = BouncerDB(db_path)
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()
        db.close()

    def test_get_pages_with_label(self, db_path):
        db = BouncerDB(db_path)
        results = [
            PageClassification("1", Classification.VISUAL, 2, 0.9, 0.05, 0.12, "Table,Figure"),
            PageClassification("2", Classification.VISUAL, 2, 0.8, 0.03, 0.08, "Picture"),
            PageClassification("3", Classification.VISUAL, 2, 0.85, 0.04, 0.10, "Table"),
            PageClassification("4", Classification.TEXT_ONLY, 1, 0.7, 0.40, None, None),
        ]
        db.save_batch(results)

        table_pages = db.get_pages_with_label("Table")
        assert table_pages == {"1", "3"}

        picture_pages = db.get_pages_with_label("Picture")
        assert picture_pages == {"2"}

        figure_pages = db.get_pages_with_label("Figure")
        assert figure_pages == {"1"}

        form_pages = db.get_pages_with_label("Form")
        assert form_pages == set()
        db.close()

    def test_get_visual_labels_for_ids(self, db_path):
        db = BouncerDB(db_path)
        results = [
            PageClassification("1", Classification.VISUAL, 2, 0.9, 0.05, 0.12, "Table,Figure"),
            PageClassification("2", Classification.VISUAL, 2, 0.8, 0.03, 0.08, "Picture"),
            PageClassification("3", Classification.TEXT_ONLY, 1, 0.7, 0.40, None, None),
        ]
        db.save_batch(results)

        labels = db.get_visual_labels_for_ids(["1", "2", "3", "999"])
        assert labels["1"] == "Table,Figure"
        assert labels["2"] == "Picture"
        assert labels["3"] is None
        assert "999" not in labels
        db.close()

    def test_get_visual_labels_empty_input(self, db_path):
        db = BouncerDB(db_path)
        labels = db.get_visual_labels_for_ids([])
        assert labels == {}
        db.close()


# ---------------------------------------------------------------------------
# Tier 1 tests
# ---------------------------------------------------------------------------


class TestTier1:
    def test_classification_key_uses_point_id_when_present(self):
        record = PageRecord(
            page_id="display_name",
            volume="TEST",
            pdf_path=TEST_PDF,
            page_index=0,
            pdf_id="display_name",
            total_pages=1,
            point_id=1234,
        )
        assert classification_key(record) == "1234"

    def test_compute_text_ratio_returns_float(self, records):
        """text_ratio should be a float between 0 and 1."""
        record = records[0]
        ratio = _compute_text_ratio(record.pdf_path, record.page_index)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_classify_tier1_high_text(self, records):
        """A page with high text ratio should be TEXT_ONLY."""
        record = records[0]
        ratio = _compute_text_ratio(record.pdf_path, record.page_index)

        # Set thresholds so this page lands in a definite bucket
        if ratio > 0:
            result = classify_tier1(record, high_threshold=ratio - 0.01, low_threshold=0.0)
            assert result.classification == Classification.TEXT_ONLY
            assert result.tier == 1

    def test_classify_tier1_low_text(self, records):
        """With thresholds set high enough, a page should be VISUAL."""
        record = records[0]
        ratio = _compute_text_ratio(record.pdf_path, record.page_index)

        result = classify_tier1(record, high_threshold=1.0, low_threshold=ratio + 0.01)
        assert result.classification == Classification.VISUAL
        assert result.tier == 1

    def test_classify_tier1_uncertain(self, records):
        """A page between thresholds should be UNCERTAIN."""
        record = records[0]
        ratio = _compute_text_ratio(record.pdf_path, record.page_index)

        if 0.01 < ratio < 0.99:
            result = classify_tier1(
                record,
                high_threshold=ratio + 0.01,
                low_threshold=ratio - 0.01,
            )
            assert result.classification == Classification.UNCERTAIN

    def test_classify_tier1_fields(self, records):
        """PageClassification should have all expected fields."""
        record = records[0]
        result = classify_tier1(record, 0.30, 0.05)
        assert result.page_id == record.page_id
        assert result.tier == 1
        assert isinstance(result.confidence, float)
        assert isinstance(result.text_ratio, float)


# ---------------------------------------------------------------------------
# Sample distribution test
# ---------------------------------------------------------------------------


class TestSample:
    def test_sample_distribution_small(self, records):
        """Sample should return sorted (page_id, ratio) tuples."""
        results = sample_distribution(records[:20], sample_size=10)
        assert len(results) <= 10
        # Should be sorted by ratio
        ratios = [r[1] for r in results]
        assert ratios == sorted(ratios)
        # Each entry is (str, float)
        for page_id, ratio in results:
            assert isinstance(page_id, str)
            assert isinstance(ratio, float)
            assert 0.0 <= ratio <= 1.0
