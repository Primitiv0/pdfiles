import torch

from pdfiles.config import Config
from pdfiles.pooling import mean_pool_embeddings, pool_batch


def test_pooling_standard_input():
    """Standard input: (1030, 128) -> (262, 128) with 2D block pooling."""
    cfg = Config()
    embedding = torch.randn(1030, 128)
    pooled = mean_pool_embeddings(embedding, cfg)

    assert pooled.shape == (262, 128)
    assert torch.isfinite(pooled).all()


def test_pooling_exact_1024_patches():
    """Input with exactly 1024 patches: (1030, 128) -> (262, 128)."""
    cfg = Config()
    embedding = torch.randn(6 + 1024, 128)  # 1030
    pooled = mean_pool_embeddings(embedding, cfg)

    assert pooled.shape == (262, 128)

    # Special tokens should be preserved exactly
    assert torch.allclose(pooled[:6], embedding[:6])


def test_pooling_extra_patches():
    """Input with more than 1024 patches should still work."""
    cfg = Config()
    embedding = torch.randn(1100, 128)
    pooled = mean_pool_embeddings(embedding, cfg)

    assert pooled.shape == (262, 128)
    assert torch.isfinite(pooled).all()


def test_pooling_fewer_patches():
    """Input with fewer than 1024 patches should pad with zeros."""
    cfg = Config()
    embedding = torch.randn(500, 128)
    pooled = mean_pool_embeddings(embedding, cfg)

    assert pooled.shape == (262, 128)
    assert torch.isfinite(pooled).all()


def test_pool_batch():
    """Pool a batch of embeddings."""
    cfg = Config()
    embeddings = [torch.randn(1030, 128) for _ in range(4)]
    pooled = pool_batch(embeddings, cfg)

    assert len(pooled) == 4
    for p in pooled:
        assert p.shape == (262, 128)


def test_block_pooling_preserves_spatial():
    """Block pooling should average 2x2 blocks from the 32x32 grid."""
    cfg = Config()

    # Create a known pattern: set all patches to 1.0, special to 0.0
    embedding = torch.zeros(1030, 128)
    embedding[6:] = 1.0  # all patches = 1.0

    pooled = mean_pool_embeddings(embedding, cfg)

    # Special tokens should be zero
    assert torch.allclose(pooled[:6], torch.zeros(6, 128))
    # Block-pooled patches should be 1.0 (mean of 2x2 blocks of 1.0)
    assert torch.allclose(pooled[6:], torch.ones(256, 128))


def test_block_pooling_2x2_mean():
    """Verify that each output block is the mean of its 2x2 input block."""
    cfg = Config()

    # Create embedding with predictable pattern
    embedding = torch.zeros(1030, 128)
    patches = embedding[6:]  # (1024, 128)

    # Set the first 2x2 block (rows 0-1, cols 0-1 of 32x32 grid) to known values
    # Grid layout: patches[row*32 + col]
    patches[0 * 32 + 0] = 1.0  # (0,0)
    patches[0 * 32 + 1] = 2.0  # (0,1)
    patches[1 * 32 + 0] = 3.0  # (1,0)
    patches[1 * 32 + 1] = 4.0  # (1,1)

    pooled = mean_pool_embeddings(embedding, cfg)

    # First block-pooled vector (index 6) should be mean of [1, 2, 3, 4] = 2.5
    expected = torch.full((128,), 2.5)
    assert torch.allclose(pooled[6], expected)
