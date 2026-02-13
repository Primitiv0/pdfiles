from pathlib import Path

import pytest
import torch

from pdfiles.config import Config
from pdfiles.embedder import Embedder
from pdfiles.renderer import render_page

TEST_PDF = Path(__file__).parent / "fixtures" / "test.pdf"


@pytest.fixture(scope="module")
def embedder():
    """Load model once for all tests in this module."""
    cfg = Config()
    return Embedder(cfg)


def test_embed_single_image(embedder):
    """Embed a single rendered page, check output shape."""
    img = render_page(TEST_PDF, 0, dpi=200)

    result = embedder.embed_images([img])

    assert len(result) == 1
    emb = result[0]
    assert emb.ndim == 2
    assert emb.shape[1] == 128, f"Expected dim=128, got {emb.shape[1]}"
    # Expect ~1030 tokens (6 special + 1024 grid)
    assert emb.shape[0] > 500, f"Too few tokens: {emb.shape[0]}"
    assert emb.shape[0] < 2000, f"Too many tokens: {emb.shape[0]}"
    assert torch.isfinite(emb).all()


def test_embed_query(embedder):
    """Embed a text query, check output shape."""
    result = embedder.embed_query("Show me handwritten documents")

    assert result.ndim == 2
    assert result.shape[1] == 128
    assert result.shape[0] > 0  # At least 1 token
    assert torch.isfinite(result).all()


def test_embed_batch(embedder):
    """Embed a batch of 2 images."""
    img0 = render_page(TEST_PDF, 0, dpi=200)
    img1 = render_page(TEST_PDF, 1, dpi=200)

    result = embedder.embed_images([img0, img1])

    assert len(result) == 2
    for emb in result:
        assert emb.shape[1] == 128
        assert emb.shape[0] > 500
