import json
import sqlite3

import numpy as np
import pytest

from pdfiles.librarian import (
    DEFAULT_CATEGORIES,
    LibrarianDB,
    STRUCTURAL_FILTERS,
    _extract_text_label,
    _get_required_surya_label,
    _reverse_search_label,
    compute_z_scores,
    load_categories,
    run_clustering,
)
from pdfiles.config import Config


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_librarian.db"


# ---------------------------------------------------------------------------
# LibrarianDB tests -- legacy cluster tables
# ---------------------------------------------------------------------------


class TestLibrarianDB:
    def test_create_tables(self, db_path):
        db = LibrarianDB(db_path)
        conn = sqlite3.connect(str(db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "page_clusters" in tables
        assert "cluster_metadata" in tables
        assert "shelf_pages" in tables
        assert "shelf_metadata" in tables
        conn.close()
        db.close()

    def test_wal_journal_mode(self, db_path):
        db = LibrarianDB(db_path)
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()
        db.close()

    def test_save_and_get_clusters(self, db_path):
        db = LibrarianDB(db_path)
        page_ids = ["1", "2", "3", "4"]
        labels = np.array([0, 1, 0, 1])
        distances = np.array([0.1, 0.2, 0.3, 0.15])

        db.save_clusters(page_ids, labels, distances)

        pages_0 = db.get_cluster_pages(0)
        assert len(pages_0) == 2
        # Should be sorted by distance
        assert pages_0[0] == ("1", 0.1)
        assert pages_0[1] == ("3", 0.3)

        pages_1 = db.get_cluster_pages(1)
        assert len(pages_1) == 2
        assert pages_1[0] == ("4", 0.15)
        assert pages_1[1] == ("2", 0.2)
        db.close()

    def test_save_clusters_replaces(self, db_path):
        """Re-running save_clusters should replace old data."""
        db = LibrarianDB(db_path)
        page_ids = ["1", "2"]
        db.save_clusters(page_ids, np.array([0, 0]), np.array([0.1, 0.2]))
        db.save_clusters(page_ids, np.array([1, 1]), np.array([0.3, 0.4]))

        pages_0 = db.get_cluster_pages(0)
        assert len(pages_0) == 0
        pages_1 = db.get_cluster_pages(1)
        assert len(pages_1) == 2
        db.close()

    def test_save_and_get_metadata(self, db_path):
        db = LibrarianDB(db_path)
        db.save_metadata(
            cluster_id=0,
            text_label="invoice, payment",
            visual_label="receipt or invoice",
            label="receipt or invoice",
            page_count=42,
            representative_ids=["1", "2", "3"],
        )

        clusters = db.get_all_clusters()
        assert len(clusters) == 1
        c = clusters[0]
        assert c["cluster_id"] == 0
        assert c["text_label"] == "invoice, payment"
        assert c["visual_label"] == "receipt or invoice"
        assert c["label"] == "receipt or invoice"
        assert c["page_count"] == 42
        assert c["representative_ids"] == ["1", "2", "3"]
        db.close()

    def test_get_page_cluster(self, db_path):
        db = LibrarianDB(db_path)
        db.save_clusters(["1", "2"], np.array([0, 1]), np.array([0.1, 0.2]))
        db.save_metadata(0, None, None, "cluster A", 1, ["1"])
        db.save_metadata(1, None, None, "cluster B", 1, ["2"])

        info = db.get_page_cluster("1")
        assert info is not None
        assert info["cluster_id"] == 0
        assert info["label"] == "cluster A"
        assert info["distance"] == pytest.approx(0.1)

        assert db.get_page_cluster("NONEXISTENT") is None
        db.close()

    def test_get_stats(self, db_path):
        db = LibrarianDB(db_path)
        page_ids = [str(i) for i in range(10)]
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        distances = np.random.rand(10)
        db.save_clusters(page_ids, labels, distances)
        db.save_metadata(0, None, None, "A", 3, [])
        db.save_metadata(1, None, None, "B", 2, [])
        db.save_metadata(2, None, None, "C", 5, [])

        stats = db.get_stats()
        assert stats["total_pages"] == 10
        assert stats["num_clusters"] == 3
        # Sorted by count descending
        assert stats["cluster_sizes"][0] == (2, 5)  # cluster 2 has 5 pages
        assert stats["cluster_sizes"][1] == (0, 3)
        assert stats["cluster_sizes"][2] == (1, 2)
        db.close()


# ---------------------------------------------------------------------------
# LibrarianDB tests -- shelf tables
# ---------------------------------------------------------------------------


class TestShelfDB:
    def test_save_and_get_shelf(self, db_path):
        db = LibrarianDB(db_path)
        pages = [
            ("1", 0.85, 4.5),
            ("2", 0.80, 3.8),
            ("3", 0.78, 3.2),
        ]
        db.save_shelf("surveillance photograph", pages, baseline_mean=0.5, baseline_std=0.08)

        shelf_pages = db.get_shelf_pages("surveillance photograph")
        assert len(shelf_pages) == 3
        # Sorted by z_score descending
        assert shelf_pages[0]["page_id"] == "1"
        assert shelf_pages[0]["z_score"] == pytest.approx(4.5)
        assert shelf_pages[2]["page_id"] == "3"
        db.close()

    def test_save_shelf_replaces(self, db_path):
        db = LibrarianDB(db_path)
        db.save_shelf("cat A", [("1", 0.8, 4.0)], 0.5, 0.1)
        db.save_shelf("cat A", [("2", 0.9, 5.0), ("3", 0.85, 4.5)], 0.5, 0.1)

        pages = db.get_shelf_pages("cat A")
        assert len(pages) == 2
        assert pages[0]["page_id"] == "2"
        db.close()

    def test_get_all_shelves_sorted_by_rank(self, db_path):
        db = LibrarianDB(db_path)
        # Save shelves with different strengths
        db.save_shelf("big shelf", [("1", 0.9, 5.0), ("2", 0.85, 4.5)], 0.5, 0.1)
        db.save_shelf("small shelf", [("3", 0.7, 3.5)], 0.5, 0.1)
        db.update_shelf_ranks()

        shelves = db.get_all_shelves()
        assert len(shelves) == 2
        # big shelf has rank 0 (higher count * mean_z)
        assert shelves[0]["category"] == "big shelf"
        assert shelves[0]["shelf_rank"] == 0
        assert shelves[1]["category"] == "small shelf"
        assert shelves[1]["shelf_rank"] == 1
        db.close()

    def test_empty_shelf_not_in_get_all(self, db_path):
        db = LibrarianDB(db_path)
        db.save_shelf("empty cat", [], 0.5, 0.1)
        shelves = db.get_all_shelves()
        assert len(shelves) == 0
        db.close()

    def test_get_all_shelved_ids(self, db_path):
        db = LibrarianDB(db_path)
        db.save_shelf("cat A", [("1", 0.8, 4.0), ("2", 0.7, 3.5)], 0.5, 0.1)
        db.save_shelf("cat B", [("2", 0.75, 3.8), ("3", 0.6, 3.2)], 0.5, 0.1)

        ids = db.get_all_shelved_ids()
        assert ids == {"1", "2", "3"}
        db.close()

    def test_shelf_stats(self, db_path):
        db = LibrarianDB(db_path)
        db.save_shelf("cat A", [("1", 0.8, 4.0), ("2", 0.7, 3.5)], 0.5, 0.1)
        db.save_shelf("cat B", [("2", 0.75, 3.8)], 0.5, 0.1)

        stats = db.get_shelf_stats()
        assert stats["shelved_pages"] == 2  # 1, 2 (2 counted once)
        assert stats["num_shelves"] == 2
        assert stats["total_assignments"] == 3  # 2 + 1
        db.close()

    def test_save_unsorted_shelf(self, db_path):
        db = LibrarianDB(db_path)
        unsorted = ["10", "11", "12"]
        db.save_unsorted_shelf(unsorted, total_unsorted=100)

        pages = db.get_shelf_pages("Unsorted / Random Discovery")
        assert len(pages) == 3

        shelves = db.get_all_shelves()
        unsorted_shelf = [s for s in shelves if s["category"] == "Unsorted / Random Discovery"]
        assert len(unsorted_shelf) == 1
        assert unsorted_shelf[0]["page_count"] == 100
        assert unsorted_shelf[0]["shelf_rank"] == 999999
        db.close()

    def test_shelf_metadata_representative_ids(self, db_path):
        db = LibrarianDB(db_path)
        pages = [
            ("1", 0.9, 5.0),
            ("2", 0.85, 4.5),
            ("3", 0.8, 4.0),
            ("4", 0.75, 3.5),
            ("5", 0.7, 3.2),
            ("6", 0.65, 3.1),
        ]
        db.save_shelf("test cat", pages, 0.5, 0.1)

        shelves = db.get_all_shelves()
        assert len(shelves) == 1
        # Representative IDs should be top 5
        assert shelves[0]["representative_ids"] == ["1", "2", "3", "4", "5"]
        db.close()

    def test_shelf_metadata_baseline(self, db_path):
        db = LibrarianDB(db_path)
        db.save_shelf("test", [("1", 0.8, 4.0)], baseline_mean=0.42, baseline_std=0.09)

        shelves = db.get_all_shelves()
        assert shelves[0]["baseline_mean"] == pytest.approx(0.42)
        assert shelves[0]["baseline_std"] == pytest.approx(0.09)
        db.close()


# ---------------------------------------------------------------------------
# Clustering tests (synthetic data)
# ---------------------------------------------------------------------------


class TestClustering:
    def test_clustering_synthetic(self, tmp_path):
        """K-Means on synthetic data with clear clusters should separate them."""
        from unittest.mock import MagicMock

        rng = np.random.RandomState(42)

        # Create 3 clearly separated clusters in 128d
        n_per_cluster = 20
        centers = rng.randn(3, 128) * 10  # well-separated
        vectors = []
        page_ids = []
        for c in range(3):
            for i in range(n_per_cluster):
                vectors.append(centers[c] + rng.randn(128) * 0.1)
                page_ids.append(f"{c:02d}{i:03d}")

        vectors = np.array(vectors, dtype=np.float32)

        # Mock the store
        store = MagicMock()
        store.export_mean_pooled_vectors.return_value = (page_ids, vectors)

        cfg = Config(num_clusters=3)
        result_ids, labels, centroids, distances, _ = run_clustering(cfg, store)

        assert len(result_ids) == 60
        assert len(labels) == 60
        assert centroids.shape == (3, 128)
        assert len(distances) == 60

        # Each original cluster should map to exactly one k-means cluster
        cluster_0_labels = set(labels[:20])
        cluster_1_labels = set(labels[20:40])
        cluster_2_labels = set(labels[40:60])

        # Each group should have a single label (perfect clustering)
        assert len(cluster_0_labels) == 1
        assert len(cluster_1_labels) == 1
        assert len(cluster_2_labels) == 1

        # All three should be different clusters
        assert len(cluster_0_labels | cluster_1_labels | cluster_2_labels) == 3

    def test_clustering_k_exceeds_points(self, tmp_path):
        """When k > n_points, should reduce k to n_points."""
        from unittest.mock import MagicMock

        vectors = np.random.randn(5, 128).astype(np.float32)
        page_ids = [str(i) for i in range(5)]

        store = MagicMock()
        store.export_mean_pooled_vectors.return_value = (page_ids, vectors)

        cfg = Config(num_clusters=100)
        result_ids, labels, centroids, distances, _ = run_clustering(cfg, store)

        assert len(result_ids) == 5
        assert centroids.shape[0] == 5  # k reduced to n_points

    def test_clustering_empty_raises(self):
        """Should raise ValueError on empty vectors."""
        from unittest.mock import MagicMock

        store = MagicMock()
        store.export_mean_pooled_vectors.return_value = ([], np.empty((0, 128)))

        cfg = Config(num_clusters=10)
        with pytest.raises(ValueError, match="No vectors found"):
            run_clustering(cfg, store)


# ---------------------------------------------------------------------------
# Labeling tests
# ---------------------------------------------------------------------------


class TestReverseSearchLabel:
    def test_best_label_selected(self):
        """Should pick the label whose embedding best matches the representative vectors."""
        rng = np.random.RandomState(42)

        # Create a "target" direction
        target = rng.randn(128).astype(np.float32)
        target /= np.linalg.norm(target)

        # Representative vectors aligned with target
        rep_vectors = np.array([target + rng.randn(128) * 0.01 for _ in range(3)], dtype=np.float32)
        rep_vectors = rep_vectors / np.linalg.norm(rep_vectors, axis=1, keepdims=True)

        # Create label embeddings: one aligned, others clearly opposite
        anti_target = -target
        label_embeddings = {
            "aligned label": np.array([target], dtype=np.float32),  # 1 token
            "random label 1": np.array([anti_target + rng.randn(128).astype(np.float32) * 0.01]),
            "random label 2": np.array([anti_target + rng.randn(128).astype(np.float32) * 0.01]),
        }

        result = _reverse_search_label(rep_vectors, label_embeddings)
        assert result == "aligned label"

    def test_empty_inputs(self):
        assert _reverse_search_label(np.empty((0, 128)), {}) is None
        assert _reverse_search_label(np.empty((0, 128)), {"a": np.zeros((1, 128))}) is None


# ---------------------------------------------------------------------------
# Z-Score tests
# ---------------------------------------------------------------------------


class TestZScores:
    def test_basic_z_scores(self):
        """Z-scores should normalize scores relative to baseline."""
        scores = [0.8, 0.9, 1.0]
        z = compute_z_scores(scores, baseline_mean=0.5, baseline_std=0.1)
        assert z[0] == pytest.approx(3.0)
        assert z[1] == pytest.approx(4.0)
        assert z[2] == pytest.approx(5.0)

    def test_z_score_zero_std(self):
        """Zero std should return all zeros (avoid division by zero)."""
        scores = [0.8, 0.9]
        z = compute_z_scores(scores, baseline_mean=0.5, baseline_std=0.0)
        assert z == [0.0, 0.0]

    def test_z_score_negative(self):
        """Scores below mean should have negative Z-scores."""
        scores = [0.3, 0.4]
        z = compute_z_scores(scores, baseline_mean=0.5, baseline_std=0.1)
        assert z[0] == pytest.approx(-2.0)
        assert z[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Category loading tests
# ---------------------------------------------------------------------------


class TestLoadCategories:
    def test_creates_default_file(self, tmp_path):
        cfg = Config(categories_file=tmp_path / "cats.txt")
        cats = load_categories(cfg)
        assert cats == DEFAULT_CATEGORIES
        assert (tmp_path / "cats.txt").exists()

    def test_reads_existing_file(self, tmp_path):
        cat_file = tmp_path / "cats.txt"
        cat_file.write_text("alpha\nbeta\n# comment\n\ngamma\n")
        cfg = Config(categories_file=cat_file)
        cats = load_categories(cfg)
        assert cats == ["alpha", "beta", "gamma"]

    def test_empty_file_uses_defaults(self, tmp_path):
        cat_file = tmp_path / "cats.txt"
        cat_file.write_text("# only comments\n\n")
        cfg = Config(categories_file=cat_file)
        cats = load_categories(cfg)
        assert cats == DEFAULT_CATEGORIES

    def test_strips_whitespace(self, tmp_path):
        cat_file = tmp_path / "cats.txt"
        cat_file.write_text("  alpha  \n  beta  \n")
        cfg = Config(categories_file=cat_file)
        cats = load_categories(cfg)
        assert cats == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Structural filter tests
# ---------------------------------------------------------------------------


class TestStructuralFilters:
    def test_get_required_surya_label(self):
        assert _get_required_surya_label("financial spreadsheet or table") == "Table"
        assert _get_required_surya_label("printed form with fields") == "Form"
        assert _get_required_surya_label("engineering diagram or blueprint") == "Figure"
        assert _get_required_surya_label("surveillance photograph") == "Picture"
        assert _get_required_surya_label("photograph of a person") == "Picture"
        assert _get_required_surya_label("graph or chart") == "Figure"
        assert _get_required_surya_label("accounting ledger") == "Table"

    def test_no_filter_for_text_categories(self):
        assert _get_required_surya_label("typed legal letter") is None
        assert _get_required_surya_label("handwritten notes") is None
        assert _get_required_surya_label("redacted or censored document") is None

    def test_case_insensitive(self):
        assert _get_required_surya_label("SURVEILLANCE PHOTOGRAPH") == "Picture"
        assert _get_required_surya_label("Financial Spreadsheet") == "Table"
