import logging

import torch
from PIL import Image

from pdfiles.config import Config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        device = self.cfg.device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        logger.info("Loading model %s on %s...", self.cfg.model_name, device)
        self._model = ColQwen2_5.from_pretrained(
            self.cfg.model_name,
            dtype=dtype,
            device_map=device,
        ).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(self.cfg.model_name)
        logger.info("Model loaded")

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """Encode images into multi-vector embeddings.

        Returns list of tensors, each shape (~1030, 128).
        """
        self._load()
        batch = self._processor.process_images(images).to(self._model.device)
        embeddings = self._model(**batch)
        # embeddings shape: (batch, seq_len, 128)
        return [embeddings[i].cpu().float() for i in range(len(images))]

    @torch.no_grad()
    def embed_query(self, query: str) -> torch.Tensor:
        """Encode text query into multi-vector embedding.

        Returns tensor of shape (num_tokens, 128).
        """
        self._load()
        batch = self._processor.process_queries([query]).to(self._model.device)
        embeddings = self._model(**batch)
        return embeddings[0].cpu().float()
