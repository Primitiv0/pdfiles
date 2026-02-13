import json
import logging
import random
import sqlite3
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from pdfiles.config import Config
from pdfiles.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Default categories -- written to categories.txt on first run.
DEFAULT_CATEGORIES = [
    "surveillance photograph",
    "handwritten notes",
    "typed legal letter",
    "financial spreadsheet or table",
    "printed form with fields",
    "map or floor plan",
    "engineering diagram or blueprint",
    "newspaper clipping",
    "photograph of a person",
    "photograph of a building or location",
    "aerial or satellite photograph",
    "medical or scientific document",
    "graph or chart",
    "receipt or invoice",
    "bank statement",
    "tax return document",
    "typed report or memo",
    "fax cover sheet",
    "envelope or mailing label",
    "business card",
    "organizational chart",
    "calendar or schedule",
    "contract or agreement",
    "court document or legal filing",
    "certificate or license",
    "identification document or ID card",
    "check or money order",
    "real estate listing or property record",
    "telephone record or call log",
    "index or table of contents",
    "blank or nearly blank page",
    "redacted or censored document",
    "computer printout or database listing",
    "handwritten letter or correspondence",
    "meeting minutes or notes",
    "affidavit or sworn statement",
    "evidence label or exhibit tag",
    "photograph of an object or evidence",
    "vehicle registration or title",
    "insurance document or policy",
    "military or government record",
    "birth or death certificate",
    "passport or visa document",
    "arrest record or criminal history",
    "laboratory report",
    "accounting ledger",
    "wire transfer record",
    "corporate filing or registration",
    "photocopy with poor quality",
    "multi-page document cover page",
]

# Structural filters: category keyword substrings -> required Surya visual labels.
# If a category name contains one of these keywords, matching pages must also have
# the corresponding Surya label detected by the Bouncer's Tier 2.
STRUCTURAL_FILTERS: dict[str, str] = {
    "spreadsheet": "Table",
    "table": "Table",
    "ledger": "Table",
    "form": "Form",
    "diagram": "Figure",
    "blueprint": "Figure",
    "chart": "Figure",
    # "photograph"/"photo" must come before "graph" to avoid false match
    "photograph": "Picture",
    "photo": "Picture",
    "graph": "Figure",
}


# ---------------------------------------------------------------------------
# Category file management
# ---------------------------------------------------------------------------

def load_categories(cfg: Config) -> list[str]:
    """Load categories from the user-editable text file.

    On first run, writes DEFAULT_CATEGORIES to the file.
    Format: one category per line, blank lines and # comments ignored.
    """
    path = cfg.categories_file

    if not path.exists():
        logger.info("Creating default categories file at %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Document categories -- one per line, # for comments\n"]
        lines.extend(f"{cat}\n" for cat in DEFAULT_CATEGORIES)
        path.write_text("".join(lines))
        return list(DEFAULT_CATEGORIES)

    categories = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        categories.append(line)

    if not categories:
        logger.warning("Categories file is empty, using defaults")
        return list(DEFAULT_CATEGORIES)

    logger.info("Loaded %d categories from %s", len(categories), path)
    return categories


def _get_required_surya_label(category: str) -> str | None:
    """Return the required Surya visual label for a category, or None."""
    cat_lower = category.lower()
    for keyword, label in STRUCTURAL_FILTERS.items():
        if keyword in cat_lower:
            return label
    return None


# ---------------------------------------------------------------------------
# LibrarianDB -- SQLite persistence (legacy cluster tables + new shelf tables)
# ---------------------------------------------------------------------------

class LibrarianDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Legacy cluster tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS page_clusters (
                page_id TEXT PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                distance_to_centroid REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_metadata (
                cluster_id INTEGER PRIMARY KEY,
                text_label TEXT,
                visual_label TEXT,
                label TEXT,
                page_count INTEGER,
                representative_ids TEXT
            )
        """)
        # New shelf tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS shelf_pages (
                category TEXT NOT NULL,
                page_id TEXT NOT NULL,
                raw_score REAL NOT NULL,
                z_score REAL NOT NULL,
                PRIMARY KEY (category, page_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS shelf_metadata (
                category TEXT PRIMARY KEY,
                page_count INTEGER NOT NULL,
                mean_z_score REAL NOT NULL,
                shelf_rank INTEGER NOT NULL,
                representative_ids TEXT,
                baseline_mean REAL,
                baseline_std REAL
            )
        """)
        self._conn.commit()

    # --- Legacy cluster methods ---

    def save_clusters(
        self,
        page_ids: list[str],
        labels: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        """Save page-cluster assignments."""
        self._conn.execute("DELETE FROM page_clusters")
        self._conn.executemany(
            "INSERT INTO page_clusters (page_id, cluster_id, distance_to_centroid) VALUES (?, ?, ?)",
            [
                (pid, int(label), float(dist))
                for pid, label, dist in zip(page_ids, labels, distances)
            ],
        )
        self._conn.commit()

    def save_metadata(
        self,
        cluster_id: int,
        text_label: str | None,
        visual_label: str | None,
        label: str,
        page_count: int,
        representative_ids: list[str],
    ) -> None:
        """Save or update metadata for a single cluster."""
        self._conn.execute(
            """INSERT OR REPLACE INTO cluster_metadata
               (cluster_id, text_label, visual_label, label, page_count, representative_ids)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cluster_id, text_label, visual_label, label, page_count, json.dumps(representative_ids)),
        )
        self._conn.commit()

    def get_cluster_pages(self, cluster_id: int) -> list[tuple[str, float]]:
        """Get (page_id, distance) for all pages in a cluster, sorted by distance."""
        rows = self._conn.execute(
            "SELECT page_id, distance_to_centroid FROM page_clusters WHERE cluster_id = ? ORDER BY distance_to_centroid",
            (cluster_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_all_clusters(self) -> list[dict]:
        """Get all cluster metadata."""
        rows = self._conn.execute(
            "SELECT cluster_id, text_label, visual_label, label, page_count, representative_ids "
            "FROM cluster_metadata ORDER BY cluster_id"
        ).fetchall()
        return [
            {
                "cluster_id": r[0],
                "text_label": r[1],
                "visual_label": r[2],
                "label": r[3],
                "page_count": r[4],
                "representative_ids": json.loads(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    def get_page_cluster(self, page_id: str) -> dict | None:
        """Get cluster info for a single page."""
        row = self._conn.execute(
            "SELECT pc.cluster_id, pc.distance_to_centroid, cm.label "
            "FROM page_clusters pc LEFT JOIN cluster_metadata cm ON pc.cluster_id = cm.cluster_id "
            "WHERE pc.page_id = ?",
            (page_id,),
        ).fetchone()
        if row is None:
            return None
        return {"cluster_id": row[0], "distance": row[1], "label": row[2]}

    def get_stats(self) -> dict:
        """Get summary statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM page_clusters").fetchone()[0]
        n_clusters = self._conn.execute("SELECT COUNT(*) FROM cluster_metadata").fetchone()[0]
        sizes = self._conn.execute(
            "SELECT cluster_id, COUNT(*) FROM page_clusters GROUP BY cluster_id ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {
            "total_pages": total,
            "num_clusters": n_clusters,
            "cluster_sizes": [(r[0], r[1]) for r in sizes],
        }

    # --- Shelf methods ---

    def save_shelf(
        self,
        category: str,
        pages: list[tuple[str, float, float]],
        baseline_mean: float,
        baseline_std: float,
    ) -> None:
        """Save a single category shelf.

        Args:
            pages: list of (page_id, raw_score, z_score) tuples
        """
        self._conn.execute("DELETE FROM shelf_pages WHERE category = ?", (category,))
        self._conn.executemany(
            "INSERT INTO shelf_pages (category, page_id, raw_score, z_score) VALUES (?, ?, ?, ?)",
            [(category, pid, float(raw), float(z)) for pid, raw, z in pages],
        )
        if pages:
            z_scores = [z for _, _, z in pages]
            mean_z = sum(z_scores) / len(z_scores)
        else:
            mean_z = 0.0

        top_ids = [pid for pid, _, _ in pages[:5]]
        self._conn.execute(
            """INSERT OR REPLACE INTO shelf_metadata
               (category, page_count, mean_z_score, shelf_rank, representative_ids, baseline_mean, baseline_std)
               VALUES (?, ?, ?, 0, ?, ?, ?)""",
            (category, len(pages), mean_z, json.dumps(top_ids), baseline_mean, baseline_std),
        )
        self._conn.commit()

    def update_shelf_ranks(self) -> None:
        """Recompute shelf_rank based on page_count * mean_z_score, descending."""
        rows = self._conn.execute(
            "SELECT category, page_count, mean_z_score FROM shelf_metadata ORDER BY (page_count * mean_z_score) DESC"
        ).fetchall()
        for rank, (cat, _, _) in enumerate(rows):
            self._conn.execute(
                "UPDATE shelf_metadata SET shelf_rank = ? WHERE category = ?",
                (rank, cat),
            )
        self._conn.commit()

    def get_all_shelves(self) -> list[dict]:
        """Get all shelf metadata, sorted by rank."""
        rows = self._conn.execute(
            "SELECT category, page_count, mean_z_score, shelf_rank, representative_ids, baseline_mean, baseline_std "
            "FROM shelf_metadata WHERE page_count > 0 ORDER BY shelf_rank"
        ).fetchall()
        return [
            {
                "category": r[0],
                "page_count": r[1],
                "mean_z_score": r[2],
                "shelf_rank": r[3],
                "representative_ids": json.loads(r[4]) if r[4] else [],
                "baseline_mean": r[5],
                "baseline_std": r[6],
            }
            for r in rows
        ]

    def get_shelf_pages(self, category: str, limit: int = 200) -> list[dict]:
        """Get pages in a shelf, sorted by z_score descending."""
        rows = self._conn.execute(
            "SELECT page_id, raw_score, z_score FROM shelf_pages WHERE category = ? ORDER BY z_score DESC LIMIT ?",
            (category, limit),
        ).fetchall()
        return [{"page_id": r[0], "raw_score": r[1], "z_score": r[2]} for r in rows]

    def get_all_shelved_ids(self) -> set[str]:
        """Get all page IDs that appear in at least one shelf."""
        rows = self._conn.execute("SELECT DISTINCT page_id FROM shelf_pages").fetchall()
        return {r[0] for r in rows}

    def get_shelf_stats(self) -> dict:
        """Get summary statistics for shelves."""
        shelved = self._conn.execute("SELECT COUNT(DISTINCT page_id) FROM shelf_pages").fetchone()[0]
        n_shelves = self._conn.execute("SELECT COUNT(*) FROM shelf_metadata WHERE page_count > 0").fetchone()[0]
        total_assignments = self._conn.execute("SELECT COUNT(*) FROM shelf_pages").fetchone()[0]
        return {
            "shelved_pages": shelved,
            "num_shelves": n_shelves,
            "total_assignments": total_assignments,
        }

    def save_unsorted_shelf(self, unsorted_ids: list[str], total_unsorted: int) -> None:
        """Save the unsorted/discovery shelf."""
        self._conn.execute("DELETE FROM shelf_pages WHERE category = 'Unsorted / Random Discovery'")
        self._conn.executemany(
            "INSERT INTO shelf_pages (category, page_id, raw_score, z_score) VALUES (?, ?, 0.0, 0.0)",
            [("Unsorted / Random Discovery", pid) for pid in unsorted_ids],
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO shelf_metadata
               (category, page_count, mean_z_score, shelf_rank, representative_ids, baseline_mean, baseline_std)
               VALUES (?, ?, 0.0, 999999, ?, 0.0, 0.0)""",
            ("Unsorted / Random Discovery", total_unsorted, json.dumps(unsorted_ids[:5])),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Legacy Clustering (kept for backwards compat)
# ---------------------------------------------------------------------------

def run_clustering(
    cfg: Config,
    store: QdrantStore,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Export vectors from Qdrant, run K-Means.

    Returns (page_ids, labels, centroids, distances_to_centroid, vectors_norm).
    """
    logger.info("Exporting vectors from Qdrant...")
    page_ids, vectors = store.export_mean_pooled_vectors()

    if len(page_ids) == 0:
        raise ValueError("No vectors found in Qdrant -- nothing to cluster")

    k = min(cfg.num_clusters, len(page_ids))
    logger.info("Clustering %d pages into %d clusters...", len(page_ids), k)

    # L2-normalize for cosine-like clustering
    vectors_norm = normalize(vectors, norm="l2")

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(vectors_norm)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Compute distance from each point to its assigned centroid
    distances = np.array([
        np.linalg.norm(vectors_norm[i] - centroids[labels[i]])
        for i in range(len(page_ids))
    ])

    logger.info("Clustering complete")
    return page_ids, labels, centroids, distances, vectors_norm


# ---------------------------------------------------------------------------
# Auto-labeling -- Strategy 1: Text extraction
# ---------------------------------------------------------------------------

def _extract_text_label(
    page_ids: list[str],
    payloads: dict[str, dict],
    top_n_words: int = 5,
) -> str | None:
    """Extract a descriptive label from the text content of representative pages."""
    stop_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at",
        "is", "it", "be", "as", "by", "was", "are", "with", "from", "that",
        "this", "not", "but", "has", "have", "had", "been", "will", "can",
        "may", "no", "all", "their", "its", "our", "your", "his", "her",
        "he", "she", "they", "we", "you", "who", "which", "what", "when",
        "where", "how", "if", "each", "than", "any", "other", "such", "per",
        "page", "date", "",
    }

    word_counter: Counter = Counter()
    pages_with_text = 0

    for pid in page_ids:
        payload = payloads.get(pid)
        if not payload:
            continue
        pdf_path = Path(payload["pdf_path"])
        page_index = payload["page_index"]
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_index]
            text = page.get_text().lower()
            doc.close()

            words = [w.strip(".,;:!?()[]{}\"'") for w in text.split()]
            words = [w for w in words if len(w) > 2 and w not in stop_words and w.isalpha()]
            if words:
                pages_with_text += 1
                word_counter.update(words)
        except Exception:
            continue

    if pages_with_text < 2 or not word_counter:
        return None

    top_words = [w for w, _ in word_counter.most_common(top_n_words)]
    return ", ".join(top_words)


# ---------------------------------------------------------------------------
# Auto-labeling -- Strategy 2: Reverse search with candidate labels
# ---------------------------------------------------------------------------

def _reverse_search_label(
    representative_vectors: np.ndarray,
    label_embeddings: dict[str, np.ndarray],
) -> str | None:
    """Find the best-matching candidate label for a cluster's representative pages."""
    if len(representative_vectors) == 0 or not label_embeddings:
        return None

    best_label = None
    best_score = -float("inf")

    for label, emb in label_embeddings.items():
        sim_matrix = representative_vectors @ emb.T
        score = sim_matrix.max(axis=1).mean()
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


# ---------------------------------------------------------------------------
# Legacy label pipeline
# ---------------------------------------------------------------------------

def label_clusters(
    cfg: Config,
    store: QdrantStore,
    db: LibrarianDB,
    page_ids: list[str],
    labels: np.ndarray,
    centroids: np.ndarray,
    distances: np.ndarray,
    vectors_norm: np.ndarray,
    embedder=None,
) -> None:
    """Label each cluster using text extraction + reverse search."""
    k = len(centroids)
    categories = load_categories(cfg)

    logger.info("Loading payloads for labeling...")
    payloads = {}
    offset = None
    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payloads[p.payload["page_id"]] = p.payload
        if next_offset is None:
            break
        offset = next_offset

    id_to_idx = {pid: i for i, pid in enumerate(page_ids)}

    label_embeddings = {}
    if embedder is not None:
        logger.info("Embedding %d candidate labels...", len(categories))
        for label_text in categories:
            emb = embedder.embed_query(label_text)
            label_embeddings[label_text] = emb.numpy()

    for cid in range(k):
        mask = labels == cid
        cluster_page_ids = [page_ids[i] for i in range(len(page_ids)) if mask[i]]
        cluster_distances = distances[mask]

        if len(cluster_page_ids) == 0:
            continue

        sorted_indices = np.argsort(cluster_distances)
        sorted_ids = [cluster_page_ids[i] for i in sorted_indices]

        n_reps = min(5, len(sorted_ids))
        representative_ids = sorted_ids[:n_reps]

        text_label = _extract_text_label(representative_ids, payloads)

        visual_label = None
        if label_embeddings:
            rep_vectors = np.array([vectors_norm[id_to_idx[pid]] for pid in representative_ids])
            visual_label = _reverse_search_label(rep_vectors, label_embeddings)

        if visual_label:
            label = visual_label
        elif text_label:
            label = text_label
        else:
            label = f"cluster {cid}"

        db.save_metadata(
            cluster_id=cid,
            text_label=text_label,
            visual_label=visual_label,
            label=label,
            page_count=len(cluster_page_ids),
            representative_ids=representative_ids,
        )
        logger.info("Cluster %d (%d pages): %s", cid, len(cluster_page_ids), label)


# ---------------------------------------------------------------------------
# Z-Score computation
# ---------------------------------------------------------------------------

def compute_z_scores(
    scores: list[float],
    baseline_mean: float,
    baseline_std: float,
) -> list[float]:
    """Compute Z-scores for a list of raw scores given baseline statistics."""
    if baseline_std < 1e-9:
        return [0.0] * len(scores)
    return [(s - baseline_mean) / baseline_std for s in scores]


# ---------------------------------------------------------------------------
# Shelf building -- the new approach
# ---------------------------------------------------------------------------

def build_shelves(
    cfg: Config,
    store: QdrantStore,
    db: LibrarianDB,
    embedder,
    bouncer_db=None,
) -> None:
    """Build category shelves using per-category search + Z-score normalization.

    For each category:
    1. Embed the category label as a query
    2. Search Qdrant for top matching pages
    3. Compute baseline score distribution from sample
    4. Z-score normalize and threshold
    5. Apply structural filters (Surya labels) where applicable
    6. Save to shelf tables
    """
    categories = load_categories(cfg)
    logger.info("Building shelves for %d categories...", len(categories))

    # Clear old shelf data
    db._conn.execute("DELETE FROM shelf_pages")
    db._conn.execute("DELETE FROM shelf_metadata")
    db._conn.commit()

    # Pre-fetch all point IDs once for random baseline sampling
    logger.info("Pre-fetching point IDs for baseline sampling...")
    id_pool = list(store.get_indexed_ids())
    logger.info("Collected %d point IDs", len(id_pool))

    for i, category in enumerate(categories):
        logger.info("[%d/%d] Processing category: %s", i + 1, len(categories), category)

        # Embed category label as a search query
        query_emb = embedder.embed_query(
            f"Find a scanned document page showing: {category}"
        )
        query_vectors = query_emb.tolist()

        # Get top-K search results
        results = store.search(query_vectors, top_k=cfg.shelf_top_k)
        if not results:
            logger.info("  No results for '%s'", category)
            continue

        # Get baseline scores from a larger sample for Z-score normalization
        baseline_scores = store.sample_random_scores(
            query_vectors, sample_size=cfg.shelf_baseline_sample, id_pool=id_pool
        )

        if len(baseline_scores) < 10:
            logger.warning("  Too few baseline scores for '%s', skipping", category)
            continue

        baseline_mean = float(np.mean(baseline_scores))
        baseline_std = float(np.std(baseline_scores))

        # Compute Z-scores for the top-K results
        raw_scores = [r.score for r in results]
        z_scores = compute_z_scores(raw_scores, baseline_mean, baseline_std)

        # Apply Z-score threshold
        candidates = [
            (r.page_id, r.score, z)
            for r, z in zip(results, z_scores)
            if z >= cfg.shelf_z_threshold
        ]

        # Apply structural filter if applicable
        required_label = _get_required_surya_label(category)
        if required_label and bouncer_db and candidates:
            candidate_ids = [pid for pid, _, _ in candidates]
            visual_labels_map = bouncer_db.get_visual_labels_for_ids(candidate_ids)
            candidates = [
                (pid, raw, z)
                for pid, raw, z in candidates
                if required_label in (visual_labels_map.get(pid) or "")
            ]

        # Sort by z_score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        db.save_shelf(category, candidates, baseline_mean, baseline_std)
        logger.info(
            "  %s: %d pages (mean_z=%.2f, baseline_mean=%.4f, baseline_std=%.4f)",
            category, len(candidates),
            sum(z for _, _, z in candidates) / len(candidates) if candidates else 0,
            baseline_mean, baseline_std,
        )

    # Rank shelves
    db.update_shelf_ranks()

    # Build unsorted shelf
    _build_unsorted_shelf(cfg, store, db, bouncer_db)

    stats = db.get_shelf_stats()
    logger.info(
        "Shelf building complete: %d unique pages across %d shelves (%d total assignments)",
        stats["shelved_pages"], stats["num_shelves"], stats["total_assignments"],
    )


def _build_unsorted_shelf(
    cfg: Config,
    store: QdrantStore,
    db: LibrarianDB,
    bouncer_db=None,
) -> None:
    """Build the unsorted/discovery shelf for pages not in any named shelf."""
    all_indexed = store.get_all_page_ids()
    shelved = db.get_all_shelved_ids()
    unsorted = all_indexed - shelved

    logger.info("Unsorted pages: %d out of %d indexed", len(unsorted), len(all_indexed))

    if not unsorted:
        return

    # Sample up to 50 pages for the display
    unsorted_list = list(unsorted)
    sample_size = min(50, len(unsorted_list))

    # If bouncer data available, prefer pages with high visual_area
    if bouncer_db:
        visual_labels_map = bouncer_db.get_visual_labels_for_ids(unsorted_list[:2000])
        # Pages with visual labels get priority
        with_labels = [pid for pid in unsorted_list if visual_labels_map.get(pid)]
        without_labels = [pid for pid in unsorted_list if not visual_labels_map.get(pid)]
        # Take from with_labels first, then without
        sample = with_labels[:sample_size]
        if len(sample) < sample_size:
            remaining = sample_size - len(sample)
            sample.extend(random.sample(without_labels, min(remaining, len(without_labels))))
    else:
        sample = random.sample(unsorted_list, sample_size)

    db.save_unsorted_shelf(sample, len(unsorted))


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def run_librarian(cfg: Config, device: str | None = None) -> None:
    """Run shelf-building pipeline: load categories -> search -> Z-score -> filter -> save."""
    store = QdrantStore(cfg)
    db = LibrarianDB(cfg.librarian_db)

    # Load embedder
    effective_device = device or cfg.device
    try:
        from pdfiles.embedder import Embedder

        label_cfg = Config(device=effective_device) if effective_device != cfg.device else cfg
        embedder = Embedder(label_cfg)
        logger.info("Loaded embedder on %s", effective_device)
    except Exception:
        logger.error("Could not load embedder -- required for shelf building")
        db.close()
        raise

    # Load bouncer DB if available
    bouncer_db = None
    try:
        from pdfiles.bouncer import BouncerDB

        if cfg.bouncer_db.exists():
            bouncer_db = BouncerDB(cfg.bouncer_db)
            logger.info("Loaded bouncer DB for structural filtering")
    except Exception:
        logger.warning("Could not load bouncer DB -- structural filters disabled")

    build_shelves(cfg, store, db, embedder, bouncer_db)

    if bouncer_db:
        bouncer_db.close()
    db.close()


def run_librarian_legacy(cfg: Config, device: str | None = None) -> None:
    """Run legacy K-Means clustering pipeline."""
    store = QdrantStore(cfg)
    db = LibrarianDB(cfg.librarian_db)

    page_ids, labels, centroids, distances, vectors_norm = run_clustering(cfg, store)
    db.save_clusters(page_ids, labels, distances)

    embedder = None
    effective_device = device or cfg.device
    try:
        from pdfiles.embedder import Embedder

        label_cfg = Config(device=effective_device) if effective_device != cfg.device else cfg
        embedder = Embedder(label_cfg)
        logger.info("Loaded embedder on %s for reverse-search labeling", effective_device)
    except Exception:
        logger.warning("Could not load embedder -- using text-only labeling")

    label_clusters(cfg, store, db, page_ids, labels, centroids, distances, vectors_norm, embedder)

    stats = db.get_stats()
    logger.info(
        "Librarian complete: %d pages in %d clusters",
        stats["total_pages"], stats["num_clusters"],
    )
    db.close()
