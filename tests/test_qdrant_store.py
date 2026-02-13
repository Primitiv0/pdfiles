import random

import pytest

from pdfiles.config import Config
from pdfiles.qdrant_store import QdrantStore


@pytest.fixture
def store():
    cfg = Config(collection_name="test_pages")
    s = QdrantStore(cfg)
    s.ensure_collection()
    yield s
    s.delete_collection()


def _rand_vecs(n_points: int, n_vecs: int = 38, dim: int = 128):
    return [
        [[random.random() for _ in range(dim)] for _ in range(n_vecs)]
        for _ in range(n_points)
    ]


def test_create_collection(store):
    """Collection should exist after ensure_collection."""
    assert store.count() == 0


def test_upsert_and_count(store):
    """Upsert 10 points and verify count."""
    point_ids = list(range(10))
    vectors = _rand_vecs(10)
    payloads = [
        {
            "page_id": f"doc{i}",
            "pdf_path": f"/path/to/doc{i}.pdf",
            "page_index": 0,
            "pdf_id": f"doc{i}",
            "volume": "VOL00011",
            "total_pages": 1,
        }
        for i in point_ids
    ]

    store.upsert_batch(point_ids, vectors, payloads)
    assert store.count() == 10


def test_search(store):
    """Upsert points and search with random query."""
    point_ids = list(range(10))
    vectors = _rand_vecs(10)
    payloads = [
        {
            "page_id": f"doc{i}",
            "pdf_path": f"/path/to/doc{i}.pdf",
            "page_index": 0,
            "pdf_id": f"doc{i}",
            "volume": "VOL00011",
            "total_pages": 1,
        }
        for i in point_ids
    ]
    store.upsert_batch(point_ids, vectors, payloads)

    query = [[random.random() for _ in range(128)] for _ in range(10)]
    results = store.search(query, top_k=5)

    assert len(results) == 5
    assert results[0].page_id.startswith("doc")
    assert results[0].point_id in set(point_ids)
    assert results[0].score > 0


def test_get_indexed_ids(store):
    """Verify resume support via get_indexed_ids."""
    point_ids = list(range(5))
    vectors = _rand_vecs(5)
    payloads = [
        {
            "page_id": f"doc{i}",
            "pdf_path": f"/path/to/doc{i}.pdf",
            "page_index": 0,
            "pdf_id": f"doc{i}",
            "volume": "VOL00011",
            "total_pages": 1,
        }
        for i in point_ids
    ]
    store.upsert_batch(point_ids, vectors, payloads)

    indexed = store.get_indexed_ids()
    assert indexed == set(point_ids)


def test_idempotent_ensure_collection(store):
    """Calling ensure_collection twice should not fail."""
    store.ensure_collection()  # Already called in fixture
    assert store.count() == 0
