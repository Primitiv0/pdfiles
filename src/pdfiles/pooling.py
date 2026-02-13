import torch

from pdfiles.config import Config


def mean_pool_embeddings(
    embedding: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Pool a single page embedding from ~1030 vectors down to 262 vectors.

    Input: (seq_len, 128) where seq_len ~ 1030
    Output: (262, 128) = 6 special tokens + 256 block-pooled grid vectors

    Strategy:
    - Keep 6 special tokens as-is
    - Treat remaining ~1024 as a 32x32 grid of patch tokens
    - Reshape into pool_grid_out x pool_grid_out blocks (16x16 = 256 blocks)
    - Mean-pool each block -> 256 vectors
    """
    n_special = cfg.num_special_tokens
    grid_side = cfg.grid_side
    grid_out = cfg.pool_grid_out

    # Split special tokens and grid patches
    special = embedding[:n_special]  # (6, 128)
    patches = embedding[n_special:]  # (~1024, 128)

    n_patches = patches.shape[0]
    expected = grid_side * grid_side  # 1024

    if n_patches >= expected:
        # Take first expected patches, reshape to grid
        grid = patches[:expected].reshape(grid_side, grid_side, -1)
    else:
        # Pad with zeros if fewer patches (shouldn't normally happen)
        padded = torch.zeros(expected, patches.shape[1])
        padded[:n_patches] = patches
        grid = padded.reshape(grid_side, grid_side, -1)

    # 2D block pooling: (32, 32, 128) -> (16, 16, 128)
    # Split each dimension into grid_out blocks of (grid_side // grid_out) patches
    block_size = grid_side // grid_out  # 32 // 16 = 2
    dim = grid.shape[-1]

    # Reshape: (32, 32, 128) -> (16, 2, 16, 2, 128)
    grid = grid.reshape(grid_out, block_size, grid_out, block_size, dim)
    # Mean-pool over both block dims (1 and 3) -> (16, 16, 128)
    block_pooled = grid.mean(dim=(1, 3))
    # Flatten spatial dims -> (256, 128)
    block_pooled = block_pooled.reshape(grid_out * grid_out, dim)

    # Concatenate: 6 special + 256 block-pooled = 262 vectors
    pooled = torch.cat([special, block_pooled], dim=0)
    return pooled


def pool_batch(
    embeddings: list[torch.Tensor],
    cfg: Config,
) -> list[torch.Tensor]:
    """Pool a batch of embeddings."""
    return [mean_pool_embeddings(e, cfg) for e in embeddings]
