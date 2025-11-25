"""
v2 Safety Assessment System

Key differences vs v1:
- Use thresholds from config (safety/confidence)
- Honor per-model checkpoints and device when available
- Add in-run embedding cache to avoid duplicate extraction during evaluation
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .embedders_v2 import create_embedder_v2, BaseEmbedderV2
from ...core.neural_classifier import SafetyClassifier  # reuse NN classifier
from ...core.ensemble import EnsembleClassifier, ModelPrediction
from ...core.safety_dimensions import (
    SafetyAssessmentResult,
    SafetyDimension,
    DimensionAnalyzer,
)
from ...config.settings import SystemConfig
from ...utils import ImageDataset


logger = logging.getLogger(__name__)


def _normalized_confidence(score: float, threshold: float) -> float:
    # Normalize distance to threshold into [0,1]
    max_dist = max(threshold, 1.0 - threshold)
    if max_dist <= 0:
        return 0.0
    return float(min(1.0, abs(score - threshold) / max_dist))


class SafetyAssessmentSystemV2:
    """Experimental v2 system with stronger config fidelity and efficiency."""

    def __init__(self, config: SystemConfig):
        self.config = config
        # Default device fallbacks if per-model device not specified
        self.default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build safety dimensions from config
        dimensions_config = {}
        if hasattr(config, "safety"):
            if isinstance(config.safety, dict) and "dimensions" in config.safety:
                dimensions_config = config.safety["dimensions"]
            elif hasattr(config.safety, "dimensions"):
                dimensions_config = config.safety.dimensions
        self.safety_dimension = SafetyDimension(dimensions_config)

        # Build embedders and classifiers per model
        self.embedders: Dict[str, BaseEmbedderV2] = {}
        self.classifiers: Dict[str, SafetyClassifier] = {}

        for m in config.models:
            if isinstance(m, dict):
                name = m.get("name")
                mtype = m.get("model_type")
                checkpoint = m.get("checkpoint")
                cache_dir = m.get("cache_dir")
                device = m.get("device") or str(self.default_device)
                embedding_dim = m.get("embedding_dim", 768)
            else:
                name = m.name
                mtype = m.model_type
                checkpoint = getattr(m, "checkpoint", None)
                cache_dir = getattr(m, "cache_dir", None)
                device = getattr(m, "device", None) or str(self.default_device)
                embedding_dim = getattr(m, "embedding_dim", 768)

            logger.info(f"[v2] Loading embedder '{name}' ({mtype}) from checkpoint: {checkpoint}")
            self.embedders[name] = create_embedder_v2(
                model_type=mtype,
                checkpoint=checkpoint,
                device=device,
                cache_dir=cache_dir,
            )

            # Batch size proxy for hidden dim, fall back to 32
            bs = 32
            if hasattr(config, "training"):
                bs = getattr(config.training, "batch_size", 32)

            self.classifiers[name] = SafetyClassifier(
                embedding_dim=embedding_dim,
                hidden_dim=bs * 8,
                num_dimensions=len(self.safety_dimension.get_all()),
                dimension_names=self.safety_dimension.get_all(),
            ).to(self.default_device)

        # Ensemble
        self.ensemble = EnsembleClassifier(
            models=config.models,
            strategy=getattr(config, "ensemble_strategy", "weighted_vote"),
        )

        # Dimension analyzer (optional/hybrid use)
        self.dimension_analyzer = DimensionAnalyzer(self.safety_dimension)

        # Performance tracking for adaptive weights (F1)
        self.model_performances: Dict[str, float] = {}

        # In-run embedding cache: (model_name, image_path) -> np.ndarray
        self._embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}

        # Thresholds
        self.safety_threshold = 0.5
        self.conf_threshold = 0.7
        if isinstance(config.safety, dict):
            self.safety_threshold = float(config.safety.get("safety_threshold", 0.5))
            self.conf_threshold = float(config.safety.get("confidence_threshold", 0.7))
        else:
            self.safety_threshold = float(getattr(config.safety, "safety_threshold", 0.5))
            self.conf_threshold = float(getattr(config.safety, "confidence_threshold", 0.7))

    def _get_embedding(self, model_name: str, image_path: str) -> np.ndarray:
        key = (model_name, image_path)
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        emb = self.embedders[model_name].extract_single_embedding(image_path)
        self._embedding_cache[key] = emb
        return emb

    def assess_image(self, image_path: str) -> SafetyAssessmentResult:
        start = time.time()

        method = getattr(self.config, "assessment_method", "ensemble")
        if method == "ensemble":
            preds: List[ModelPrediction] = []
            for model_name, embedder in self.embedders.items():
                emb = self._get_embedding(model_name, image_path)
                x = torch.tensor(emb, dtype=torch.float32, device=self.default_device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                clf = self.classifiers[model_name]
                clf.eval()
                with torch.no_grad():
                    overall, dim_scores = clf(x)
                    score = float(overall.item())
                    dims = {d: float(s.item()) for d, s in dim_scores.items()}
                    preds.append(
                        ModelPrediction(
                            model_name=model_name,
                            is_safe=score > self.safety_threshold,
                            safety_score=score,
                            confidence=_normalized_confidence(score, self.safety_threshold),
                            dimension_scores=dims,
                            embedding=emb,
                        )
                    )
            result = self.ensemble.predict(preds)
            result.image_path = image_path
        else:
            # single model: choose the first defined in config
            if isinstance(self.config.models[0], dict):
                model_name = self.config.models[0]["name"]
            else:
                model_name = self.config.models[0].name
            emb = self._get_embedding(model_name, image_path)
            x = torch.tensor(emb, dtype=torch.float32, device=self.default_device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            clf = self.classifiers[model_name]
            clf.eval()
            with torch.no_grad():
                overall, dim_scores = clf(x)
                score = float(overall.item())
                dims = {d: float(s.item()) for d, s in dim_scores.items()}
                result = SafetyAssessmentResult(
                    image_path=image_path,
                    overall_safety_score=score,
                    is_safe=score > self.safety_threshold,
                    dimension_scores=dims,
                    confidence=_normalized_confidence(score, self.safety_threshold),
                    method_used=method,
                    model_name=model_name,
                    processing_time=0.0,
                )

        result.processing_time = time.time() - start
        return result

    def evaluate_dataset(self, dataset: ImageDataset) -> Dict:
        all_results: List[SafetyAssessmentResult] = []
        all_labels: List[bool] = []

        model_predictions: Dict[str, List[bool]] = {n: [] for n in self.embedders.keys()}
        model_labels: Dict[str, List[bool]] = {n: [] for n in self.embedders.keys()}

        for idx in range(len(dataset)):
            _, label, path = dataset[idx]
            res = self.assess_image(path)
            all_results.append(res)
            all_labels.append(bool(label["is_safe"]))

            if getattr(self.config, "assessment_method", "ensemble") == "ensemble":
                for model_name in self.embedders.keys():
                    emb = self._get_embedding(model_name, path)
                    x = torch.tensor(emb, dtype=torch.float32, device=self.default_device)
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    clf = self.classifiers[model_name]
                    clf.eval()
                    with torch.no_grad():
                        overall, _ = clf(x)
                        score = float(overall.item())
                        model_predictions[model_name].append(score > self.safety_threshold)
                        model_labels[model_name].append(bool(label["is_safe"]))

        ensemble_metrics = self._metrics([r.is_safe for r in all_results], all_labels)

        individual_metrics: Dict[str, Dict] = {}
        for model_name in self.embedders.keys():
            if model_predictions[model_name]:
                individual_metrics[model_name] = self._metrics(model_predictions[model_name], model_labels[model_name])

        for model_name, metrics in individual_metrics.items():
            self.model_performances[model_name] = metrics.get("f1_score", 0.0)
        if hasattr(self.ensemble, "update_weights"):
            self.ensemble.update_weights(self.model_performances)

        best_model = None
        if individual_metrics:
            best_model = max(individual_metrics.items(), key=lambda kv: kv[1].get("f1_score", 0.0))[0]

        return {
            "ensemble_metrics": ensemble_metrics,
            "individual_metrics": individual_metrics,
            "model_performances": self.model_performances,
            "best_individual_model": best_model,
            "ensemble_improvement": (
                ensemble_metrics.get("f1_score", 0.0)
                - max((m.get("f1_score", 0.0) for m in individual_metrics.values()), default=0.0)
            ),
        }

    @staticmethod
    def _metrics(predictions: List[bool], labels: List[bool]) -> Dict:
        import numpy as np

        p = np.asarray(predictions)
        y = np.asarray(labels)
        tp = int(((p == True) & (y == True)).sum())
        tn = int(((p == False) & (y == False)).sum())
        fp = int(((p == True) & (y == False)).sum())
        fn = int(((p == False) & (y == True)).sum())

        total = tp + tn + fp + fn
        accuracy = float((tp + tn) / total) if total else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

