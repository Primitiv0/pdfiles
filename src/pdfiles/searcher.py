import logging
from collections import defaultdict

from pdfiles.config import Config
from pdfiles.embedder import Embedder
from pdfiles.qdrant_store import QdrantStore, SearchResult
from pdfiles.renderer import render_page

logger = logging.getLogger(__name__)


# Default query variants for multi-query RRF
QUERY_VARIANTS = [
    "{query}",
    "Photograph or image showing: {query}",
    "A document page with {query}",
]

# RRF constant (standard value from literature)
RRF_K = 60


class Searcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = Embedder(cfg)
        self.store = QdrantStore(cfg)

    @staticmethod
    def expand_query(query: str) -> str:
        """Expand short queries for better MaxSim discrimination.

        Short queries (<=3 words) produce very few token vectors, making
        the MaxSim score landscape flat. Wrapping them in descriptive
        context gives ColQwen2.5 more tokens to work with.
        """
        words = query.strip().split()
        if len(words) <= 3:
            return f"Photograph or image showing: {query}"
        return query

    def search_by_image(self, image: "PIL.Image.Image", top_k: int = 10) -> list[SearchResult]:
        """Search for pages visually similar to an uploaded image."""
        embeddings = self.embedder.embed_images([image])
        query_vectors = embeddings[0].tolist()
        return self.store.search(query_vectors, top_k=top_k)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for pages matching a text query (single-query with expansion)."""
        expanded = self.expand_query(query)
        if expanded != query:
            logger.info("Expanded query: %r -> %r", query, expanded)
        query_emb = self.embedder.embed_query(expanded)
        query_vectors = query_emb.tolist()
        return self.store.search(query_vectors, top_k=top_k)

    def search_multi(
        self,
        query: str,
        top_k: int = 10,
        variants: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search with multiple query variants, merge by reciprocal rank fusion.

        Runs N phrasings of the query independently, then combines results
        using RRF scoring: score = sum(1 / (K + rank_i)) across all variants.
        This produces more robust rankings than any single phrasing.

        Args:
            query: Raw user query
            top_k: Number of final results to return
            variants: Optional custom variant templates (must contain {query}).
                      Defaults to QUERY_VARIANTS.
        """
        if variants is None:
            variants = QUERY_VARIANTS

        # Fetch more per-variant to give RRF enough candidates
        per_variant_k = top_k * 3

        # Collect results per variant
        all_variant_results: list[list[SearchResult]] = []
        for template in variants:
            variant_query = template.format(query=query)
            query_emb = self.embedder.embed_query(variant_query)
            query_vectors = query_emb.tolist()
            results = self.store.search(query_vectors, top_k=per_variant_k)
            all_variant_results.append(results)
            logger.debug(
                "Variant %r returned %d results (top=%.4f)",
                template,
                len(results),
                results[0].score if results else 0,
            )

        # Reciprocal Rank Fusion (keyed by point_id to distinguish pages)
        rrf_scores: dict[int, float] = defaultdict(float)
        best_result: dict[int, SearchResult] = {}

        for variant_results in all_variant_results:
            for rank, result in enumerate(variant_results):
                rrf_scores[result.point_id] += 1.0 / (RRF_K + rank + 1)
                # Keep the result object with the highest original score
                if result.point_id not in best_result or result.score > best_result[result.point_id].score:
                    best_result[result.point_id] = result

        # Sort by RRF score descending, take top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(
                page_id=best_result[pid].page_id,
                pdf_path=best_result[pid].pdf_path,
                page_index=best_result[pid].page_index,
                pdf_id=best_result[pid].pdf_id,
                volume=best_result[pid].volume,
                total_pages=best_result[pid].total_pages,
                score=rrf_score,
                point_id=pid,
            )
            for pid, rrf_score in ranked
        ]

    def render_result(self, result: SearchResult) -> "PIL.Image.Image":
        """Render the page image for a search result."""
        return render_page(
            self.cfg.resolve_pdf_path(result.pdf_path),
            result.page_index,
            dpi=self.cfg.render_dpi,
        )
