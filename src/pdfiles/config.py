import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Data paths
    data_root: Path = field(default_factory=lambda: Path(
        os.environ.get("DATA_ROOT", "/data")
    ))
    opt_file: Path = field(default=None)
    images_root: Path = field(default=None)

    # Rendering
    render_dpi: int = 200

    # Model
    model_name: str = "vidore/colqwen2.5-v0.2"
    device: str = field(default_factory=lambda: os.environ.get(
        "DEVICE", "cuda:0"
    ))
    batch_size: int = 4

    # Pooling
    use_pooling: bool = True
    num_special_tokens: int = 6
    grid_side: int = 32  # 32x32 = 1024 grid patches
    pool_grid_out: int = 16  # output grid dimension (16x16 = 256 block-pooled)
    pooled_vectors: int = 262  # 6 special + 256 block-pooled (16x16)

    # Qdrant
    qdrant_url: str = field(default_factory=lambda: os.environ.get(
        "QDRANT_URL", "http://localhost:6335"
    ))
    collection_name: str = "pages"
    vector_dim: int = 128

    # Workers
    num_render_workers: int = 11

    # Bouncer (page classification)
    bouncer_db: Path = field(default=None)
    bouncer_high_threshold: float = 0.30
    bouncer_low_threshold: float = 0.05
    bouncer_visual_area_threshold: float = 0.05

    # Manifest (scan persistence)
    manifest_db: Path = field(default=None)

    # Librarian (clustering / shelves)
    librarian_db: Path = field(default=None)
    num_clusters: int = 30
    categories_file: Path = field(default=None)
    shelf_z_threshold: float = 3.0
    shelf_top_k: int = 200
    shelf_baseline_sample: int = 1000

    # Admin mode
    admin_mode: bool = field(default_factory=lambda: os.environ.get(
        "ADMIN_MODE", "true"
    ).lower() == "true")

    def __post_init__(self):
        if self.opt_file is None:
            self.opt_file = self.data_root / "DATA" / "VOL00011.OPT"
        if self.images_root is None:
            self.images_root = self.data_root / "IMAGES"
        if self.manifest_db is None:
            self.manifest_db = Path(os.environ.get(
                "MANIFEST_DB", str(self.data_root / "manifest.db")
            ))
        if self.bouncer_db is None:
            self.bouncer_db = Path(os.environ.get(
                "BOUNCER_DB", str(self.data_root / "bouncer.db")
            ))
        if self.librarian_db is None:
            self.librarian_db = Path(os.environ.get(
                "LIBRARIAN_DB", str(self.data_root / "librarian.db")
            ))
        if self.categories_file is None:
            self.categories_file = Path(os.environ.get(
                "CATEGORIES_FILE", str(self.data_root / "categories.txt")
            ))
        # Allow overriding HuggingFace cache location
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)

        # Warn if data paths don't exist
        if not self.data_root.exists():
            logger.warning("Data root does not exist: %s", self.data_root)
        elif not self.images_root.exists():
            logger.warning("No IMAGES/ directory found in %s", self.data_root)

    def resolve_pdf_path(self, stored_path: str) -> Path:
        """Resolve a stored PDF path against the current data_root.

        Handles three formats:
        - Relative paths (new): "00290498.pdf" -> data_root / name
        - OPT-indexed absolute: ".../IMAGES/0330/file.pdf" -> data_root / IMAGES/...
        - Walk-indexed absolute: "/data/00290498.pdf" -> data_root / name
        """
        p = Path(stored_path)

        if not p.is_absolute():
            return self.data_root / p

        if p.exists():
            return p

        # OPT-style: extract from IMAGES/ onward
        marker = "/IMAGES/"
        idx = stored_path.find(marker)
        if idx != -1:
            relative = stored_path[idx + 1:]  # "IMAGES/0330/file.pdf"
            candidate = self.data_root / relative
            if candidate.exists():
                return candidate

        # Walk-style: just the filename under data_root
        candidate = self.data_root / p.name
        if candidate.exists():
            return candidate

        return p
