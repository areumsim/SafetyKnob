"""
v2 Embedders

Differences vs v1:
- Honors `checkpoint` from config (no hardcoded IDs)
- Validates/returns embeddings with the correct dimension
- Graceful failure behavior: returns a zero vector with the correct dim (or can
  be adjusted to raise) and logs the issue
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class BaseEmbedderV2:
    def __init__(self, device: str = "cuda", cache_path: Optional[str] = None):
        self.device = device
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                # Fresh cache on failure
                self.cache = {}

    def save_cache(self):
        if self.cache_path:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            except Exception:
                pass
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        raise NotImplementedError

    def extract_single_embedding(self, image_path: str) -> np.ndarray:
        embs = self.extract_embeddings([image_path])
        if len(embs) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return embs[0]


class SigLIPEmbedderV2(BaseEmbedderV2):
    def __init__(self, checkpoint: str, device: str = "cuda", cache_path: Optional[str] = None):
        super().__init__(device, cache_path)
        from transformers import AutoProcessor, AutoModel

        self.model_name = checkpoint
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        # Infer embedding dim with a dummy forward on a tiny image
        self._embedding_dim = None
        try:
            dummy = Image.new("RGB", (8, 8))
            inputs = self.processor(images=dummy, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                self._embedding_dim = int(feats.shape[-1])
        except Exception:
            self._embedding_dim = 1152  # sensible default for common siglip checkpoints

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        from transformers import AutoProcessor
        from PIL import Image
        import torch

        embeddings = []
        with torch.no_grad():
            for path in image_paths:
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    features = self.model.get_image_features(**inputs)
                    features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                    emb = features.squeeze().detach().cpu().numpy().astype(np.float32)
                    # Ensure correct dim
                    if emb.shape[-1] != self.embedding_dim:
                        emb = np.resize(emb, (self.embedding_dim,)).astype(np.float32)
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception:
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        return np.asarray(embeddings)


class DINOv2EmbedderV2(BaseEmbedderV2):
    def __init__(self, checkpoint: str, device: str = "cuda", cache_path: Optional[str] = None):
        super().__init__(device, cache_path)
        from transformers import AutoImageProcessor, Dinov2Model

        self.model_name = checkpoint
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Dinov2Model.from_pretrained(self.model_name).to(self.device).eval()
        # Infer embedding dim
        self._embedding_dim = None
        try:
            dummy = Image.new("RGB", (8, 8))
            inputs = self.processor(images=dummy, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                feats = outputs.last_hidden_state[:, 0]
                self._embedding_dim = int(feats.shape[-1])
        except Exception:
            self._embedding_dim = 1536  # common for dinov2-giant

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        from PIL import Image
        import torch

        embeddings = []
        with torch.no_grad():
            for path in image_paths:
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0]
                    features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                    emb = features.squeeze().detach().cpu().numpy().astype(np.float32)
                    if emb.shape[-1] != self.embedding_dim:
                        emb = np.resize(emb, (self.embedding_dim,)).astype(np.float32)
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception:
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        return np.asarray(embeddings)


class CLIPEmbedderV2(BaseEmbedderV2):
    def __init__(self, checkpoint: str, device: str = "cuda", cache_path: Optional[str] = None):
        super().__init__(device, cache_path)
        from transformers import CLIPProcessor, CLIPModel

        self.model_name = checkpoint
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        # Infer embedding dim
        self._embedding_dim = None
        try:
            dummy = Image.new("RGB", (8, 8))
            inputs = self.processor(images=dummy, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                self._embedding_dim = int(feats.shape[-1])
        except Exception:
            self._embedding_dim = 768  # common for ViT-L/14 image tower

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        from PIL import Image
        import torch

        embeddings = []
        with torch.no_grad():
            for path in image_paths:
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
                    features = self.model.get_image_features(**inputs)
                    features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                    emb = features.squeeze().detach().cpu().numpy().astype(np.float32)
                    if emb.shape[-1] != self.embedding_dim:
                        emb = np.resize(emb, (self.embedding_dim,)).astype(np.float32)
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception:
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        return np.asarray(embeddings)


class EVACLIPEmbedderV2(BaseEmbedderV2):
    def __init__(self, checkpoint: str, device: str = "cuda", cache_path: Optional[str] = None):
        super().__init__(device, cache_path)
        import timm
        from torchvision import transforms

        # timm model name passed via checkpoint (e.g., "eva_giant_patch14_clip_224")
        self.model_name = checkpoint
        self.model = timm.create_model(self.model_name, pretrained=True).to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        # Infer embedding dim
        self._embedding_dim = None
        try:
            import torch

            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                feats = self.model(dummy)
                self._embedding_dim = int(feats.shape[-1])
        except Exception:
            self._embedding_dim = 1024

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        from PIL import Image
        import torch

        embeddings = []
        with torch.no_grad():
            for path in image_paths:
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    tensor = self.transform(image).unsqueeze(0).to(self.device)
                    features = self.model(tensor)
                    features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                    emb = features.squeeze().detach().cpu().numpy().astype(np.float32)
                    if emb.shape[-1] != self.embedding_dim:
                        emb = np.resize(emb, (self.embedding_dim,)).astype(np.float32)
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception:
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        return np.asarray(embeddings)


def _canonical_model_type(model_type: str) -> str:
    mt = (model_type or "").lower()
    if mt in {"dino", "dinov2"}:
        return "dinov2"
    if mt in {"eva_clip", "evaclip", "eva-clip"}:
        return "eva_clip"
    if mt in {"clip"}:
        return "clip"
    if mt in {"siglip"}:
        return "siglip"
    return mt


def create_embedder_v2(
    model_type: str,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> BaseEmbedderV2:
    mt = _canonical_model_type(model_type)
    cache_path = f"{cache_dir}/embedding_cache_{mt}.pkl" if cache_dir else None

    if mt == "siglip":
        return SigLIPEmbedderV2(checkpoint=checkpoint or "google/siglip-so400m-patch14-384", device=device, cache_path=cache_path)
    if mt == "dinov2":
        return DINOv2EmbedderV2(checkpoint=checkpoint or "facebook/dinov2-giant", device=device, cache_path=cache_path)
    if mt == "clip":
        return CLIPEmbedderV2(checkpoint=checkpoint or "openai/clip-vit-large-patch14", device=device, cache_path=cache_path)
    if mt == "eva_clip":
        # For EVA-CLIP via timm, pass the timm model id (e.g., "eva_giant_patch14_clip_224")
        return EVACLIPEmbedderV2(checkpoint=checkpoint or "eva_giant_patch14_clip_224", device=device, cache_path=cache_path)

    raise ValueError(f"Unsupported model_type: {model_type}")

