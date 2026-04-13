# embedders.py

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from torchvision import transforms
from tqdm import tqdm
import pickle
import timm


# ================================
# 💡 공통 Embedder Base Class
# ================================
class BaseEmbedder:
    def __init__(self, device="cuda", cache_path=None):
        self.device = device
        self.cache_path = cache_path
        self.cache = {}

        if cache_path:
            try:
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"✅ 임베딩 캐시 로드됨: {len(self.cache)}개")
            except:
                print("⚠️ 캐시 없음 또는 로드 실패. 새로 생성됩니다.")

    def save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"💾 캐시 저장 완료: {self.cache_path}")

    def extract_embeddings(self, image_paths):
        raise NotImplementedError("이 메서드는 서브 클래스에서 구현해야 합니다.")
    
    def extract_single_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding for a single image"""
        embeddings = self.extract_embeddings([image_path])
        if len(embeddings) > 0:
            return embeddings[0]
        else:
            # Return zero vector with appropriate dimension
            # This should be overridden by subclasses with correct embedding_dim
            import logging
            logging.warning(f"Failed to extract embedding for {image_path}, returning zero vector")
            return np.zeros(768)  # Default, subclasses should override


# ================================
# 📦 SigLIP Embedder
# ================================
class SigLIPEmbedder(BaseEmbedder):
    def __init__(self, device="cuda", cache_path=None, checkpoint=None):
        super().__init__(device, cache_path)
        self.model_name = checkpoint or "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def extract_embeddings(self, image_paths):
        embeddings = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="SigLIP Embedding"):
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(
                        self.device
                    )
                    features = self.model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                    emb = features.squeeze().cpu().numpy()
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception as e:
                    print(f"❌ 오류 발생 [{path}]: {e}")
                    continue
        return np.array(embeddings)


# ================================
# 📦 DINOv2 Embedder
# ================================
class DINOEmbedder(BaseEmbedder):
    def __init__(self, device="cuda", cache_path=None, checkpoint=None):
        super().__init__(device, cache_path)
        from transformers import AutoImageProcessor, Dinov2Model, AutoModel

        self.model_name = checkpoint or "facebook/dinov2-large"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Dinov2Model.from_pretrained(self.model_name).to(self.device).eval()
        # self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def extract_embeddings(self, image_paths):
        embeddings = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="DINOv2 Embedding"):
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(
                        self.device
                    )
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0]  # [CLS] 토큰 사용
                    features = features / features.norm(dim=-1, keepdim=True)
                    emb = features.squeeze().cpu().numpy()
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception as e:
                    print(f"❌ 오류 발생 [{path}]: {e}")
                    continue
        return np.array(embeddings)


# ================================
# 📦 CLIP Embedder
# ================================
class CLIPEmbedder(BaseEmbedder):
    def __init__(self, device="cuda", cache_path=None, checkpoint=None):
        super().__init__(device, cache_path)
        from transformers import CLIPProcessor, CLIPModel

        self.model_name = checkpoint or "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()

    def extract_embeddings(self, image_paths):
        embeddings = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="CLIP Embedding"):
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.processor(
                        images=image, return_tensors="pt", padding=True
                    ).to(self.device)
                    features = self.model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                    emb = features.squeeze().cpu().numpy()
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception as e:
                    print(f"❌ 오류 발생 [{path}]: {e}")
                    continue
        return np.array(embeddings)


class EVACLIPEmbedder(BaseEmbedder):
    def __init__(self, device="cuda", cache_path=None, checkpoint=None):
        super().__init__(device, cache_path)

        # "timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k" :  "eva_giant_patch14_clip_224"
        self.model_name = checkpoint or "timm/eva_giant_patch14_clip_224"
        self.model = (
            timm.create_model(self.model_name, pretrained=True).to(self.device).eval()
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def extract_embeddings(self, image_paths):
        embeddings = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="EVA-CLIP Embedding"):
                if path in self.cache:
                    embeddings.append(self.cache[path])
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    tensor = self.transform(image).unsqueeze(0).to(self.device)
                    features = self.model(tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    emb = features.squeeze().cpu().numpy()
                    embeddings.append(emb)
                    self.cache[path] = emb
                except Exception as e:
                    print(f"❌ 오류 발생 [{path}]: {e}")
                    continue
        return np.array(embeddings)


def get_embedder(config):
    checkpoint = config.get("checkpoint", None)
    if config["model_type"] == "siglip":
        return SigLIPEmbedder(device=config["device"], cache_path=config["cache_path"], checkpoint=checkpoint)
    elif config["model_type"] in ["dino", "dinov2"]:
        return DINOEmbedder(device=config["device"], cache_path=config["cache_path"], checkpoint=checkpoint)
    elif config["model_type"] == "clip":
        return CLIPEmbedder(device=config["device"], cache_path=config["cache_path"], checkpoint=checkpoint)
    elif config["model_type"] == "eva_clip":
        return EVACLIPEmbedder(device=config["device"], cache_path=config["cache_path"], checkpoint=checkpoint)

    else:
        raise ValueError(f"❌ 지원하지 않는 모델: {config['model_type']}")


def create_embedder(model_type: str, device: str = "cuda", cache_dir: str = None, checkpoint: str = None) -> BaseEmbedder:
    """Create embedder instance based on model type"""
    config = {
        "model_type": model_type,
        "device": device,
        "cache_path": f"{cache_dir}/embedding_cache_{model_type}.pkl" if cache_dir else None,
        "checkpoint": checkpoint
    }
    return get_embedder(config)
