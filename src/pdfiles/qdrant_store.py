import logging
import random
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models

from pdfiles.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    page_id: str
    pdf_path: str
    page_index: int
    pdf_id: str
    volume: str
    total_pages: int
    score: float
    point_id: int = 0


class QdrantStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = QdrantClient(url=cfg.qdrant_url)
        self.collection = cfg.collection_name

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            logger.info("Collection '%s' already exists", self.collection)
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.cfg.vector_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                on_disk=True,
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True),
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )
        logger.info("Created collection '%s'", self.collection)

    def get_indexed_ids(self) -> set[int]:
        """Get all point IDs already in the collection (for resume support)."""
        indexed = set()
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection,
                limit=10000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            for p in points:
                indexed.add(p.id)
            if next_offset is None:
                break
            offset = next_offset
        return indexed

    def upsert_batch(
        self,
        point_ids: list[int],
        vectors: list[list[list[float]]],
        payloads: list[dict],
    ) -> None:
        """Upsert a batch of multi-vector points.

        Args:
            point_ids: List of integer Qdrant point IDs
            vectors: List of multi-vectors, each is list[list[float]] shape (N, 128)
            payloads: List of payload dicts
        """
        points = [
            models.PointStruct(
                id=pid,
                vector=vecs,
                payload=payload,
            )
            for pid, vecs, payload in zip(point_ids, vectors, payloads)
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(
        self,
        query_vectors: list[list[float]],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search with multi-vector query (MaxSim scoring)."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vectors,
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                page_id=p.payload["page_id"],
                pdf_path=p.payload["pdf_path"],
                page_index=p.payload["page_index"],
                pdf_id=p.payload["pdf_id"],
                volume=p.payload["volume"],
                total_pages=p.payload["total_pages"],
                score=p.score,
                point_id=p.id,
            )
            for p in result.points
        ]

    def export_mean_pooled_vectors(self) -> tuple[list[str], np.ndarray]:
        """Export all vectors, mean-pooling multi-vectors to 1x128d per page.

        Returns (page_ids, vectors) where vectors is shape (N, 128).
        """
        page_ids = []
        vectors = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=["page_id"],
                with_vectors=True,
            )
            for p in points:
                page_ids.append(p.payload["page_id"])
                # p.vector is list[list[float]] — mean-pool to (128,)
                vectors.append(np.mean(p.vector, axis=0))

            if next_offset is None:
                break
            offset = next_offset

        logger.info("Exported %d vectors from Qdrant", len(page_ids))
        return page_ids, np.array(vectors, dtype=np.float32) if vectors else np.empty((0, self.cfg.vector_dim), dtype=np.float32)

    def get_all_page_ids(self) -> set[str]:
        """Get all page IDs in the collection."""
        page_ids = set()
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=10000,
                offset=offset,
                with_payload=["page_id"],
                with_vectors=False,
            )
            for p in points:
                page_ids.add(p.payload["page_id"])
            if next_offset is None:
                break
            offset = next_offset
        return page_ids

    def sample_random_scores(
        self,
        query_vectors: list[list[float]],
        sample_size: int = 1000,
        id_pool: list[int] | None = None,
    ) -> list[float]:
        """Score a random sample of points against a query for baseline computation.

        Uses scroll to collect point IDs, randomly samples, then scores only
        those points via HasIdCondition filter to get a true random baseline.

        Args:
            id_pool: Pre-fetched list of all point IDs (avoids repeated full scans).
                     If None, fetches all IDs from the collection.
        """
        if id_pool is None:
            id_pool = list(self.get_indexed_ids())

        if not id_pool:
            return []

        sampled_ids = random.sample(id_pool, min(sample_size, len(id_pool)))

        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vectors,
            query_filter=models.Filter(
                must=[models.HasIdCondition(has_id=sampled_ids)]
            ),
            limit=sample_size,
            with_payload=False,
        )
        return [p.score for p in result.points]

    def search_similar(self, point_id: int, top_k: int = 10) -> list[SearchResult]:
        """Find pages similar to a given page using its stored embeddings."""
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id],
            with_payload=False,
            with_vectors=True,
        )
        if not points:
            return []
        query_vectors = points[0].vector
        results = self.search(query_vectors, top_k=top_k + 1)
        return [r for r in results if r.point_id != point_id][:top_k]

    def count(self) -> int:
        """Get number of indexed points."""
        info = self.client.get_collection(self.collection)
        return info.points_count

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection)
