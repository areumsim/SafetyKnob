#!/usr/bin/env python3
"""
Benchmark script for SafetyKnob v2 pipeline (side-by-side with v1 optional).

- Measures: latency per image, throughput, optional accuracy/F1 if labels given,
  and (if CUDA) peak memory usage.
- Does not modify v1; can optionally compare against v1 for reference.

Usage examples:
  python scripts/benchmark_v2.py --data-dir ./data/test --patterns "*.jpg,*.png" \
      --output ./results/bench_v2.json

  python scripts/benchmark_v2.py --data-dir ./data/test --labels ./data/labels.json \
      --compare-v1 --output ./results/bench_v1_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import SystemConfig
from src_v2.core import SafetyAssessmentSystemV2


def load_config(path: Path) -> SystemConfig:
    if path.exists():
        with open(path, "r") as f:
            cfg = json.load(f)
        return SystemConfig.from_dict(cfg)
    return SystemConfig()


def collect_images(root: Path, patterns: List[str], recursive: bool = True) -> List[str]:
    files: List[str] = []
    if recursive:
        for pat in patterns:
            files.extend(str(p) for p in root.rglob(pat))
    else:
        for pat in patterns:
            files.extend(str(p) for p in root.glob(pat))
    return files


def safe_read_json(path: Optional[Path]) -> Dict:
    if not path:
        return {}
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def metrics_binary(preds: List[bool], labels: List[bool]) -> Dict[str, float]:
    p = np.asarray(preds)
    y = np.asarray(labels)
    tp = int(((p == True) & (y == True)).sum())
    tn = int(((p == False) & (y == False)).sum())
    fp = int(((p == True) & (y == False)).sum())
    fn = int(((p == False) & (y == True)).sum())
    total = tp + tn + fp + fn
    acc = float((tp + tn) / total) if total else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def benchmark_system(name: str, system, images: List[str], labels_map: Dict[str, Dict]) -> Dict:
    import torch

    # Try to capture CUDA memory
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    results = []
    t0 = time.time()
    for i, img in enumerate(images, 1):
        t_img0 = time.time()
        res = system.assess_image(img)
        t_img1 = time.time()
        results.append({
            "image": img,
            "is_safe": bool(res.is_safe),
            "score": float(res.overall_safety_score),
            "confidence": float(res.confidence),
            "time": float(t_img1 - t_img0),
        })
    t1 = time.time()

    times = [r["time"] for r in results]
    total_time = float(t1 - t0)
    avg = float(np.mean(times)) if times else 0.0
    p50 = float(np.percentile(times, 50)) if times else 0.0
    p90 = float(np.percentile(times, 90)) if times else 0.0
    p99 = float(np.percentile(times, 99)) if times else 0.0
    throughput = float(len(images) / total_time) if total_time > 0 else 0.0

    # Accuracy metrics if labels provided
    preds, gts = [], []
    if labels_map:
        for r in results:
            img_path = r["image"]
            lab = labels_map.get(img_path) or labels_map.get(str(Path(img_path)))
            if lab and isinstance(lab, dict) and "is_safe" in lab:
                preds.append(bool(r["is_safe"]))
                gts.append(bool(lab["is_safe"]))
    acc = metrics_binary(preds, gts) if preds and gts else None

    peak_mem = None
    if cuda_available:
        try:
            peak_mem = int(torch.cuda.max_memory_allocated())
        except Exception:
            peak_mem = None

    return {
        "name": name,
        "count": len(images),
        "total_time": total_time,
        "avg_time": avg,
        "p50_time": p50,
        "p90_time": p90,
        "p99_time": p99,
        "throughput": throughput,
        "accuracy": acc,
        "peak_cuda_mem": peak_mem,
    }


def main():
    ap = argparse.ArgumentParser("Benchmark v2 vs v1 (optional)")
    ap.add_argument("--data-dir", required=True, help="Directory with images")
    ap.add_argument("--labels", help="Optional labels.json for accuracy metrics")
    ap.add_argument("--config", default="config.json", help="Config JSON path")
    ap.add_argument("--patterns", default="*.jpg,*.jpeg,*.png", help="Comma-separated glob patterns")
    ap.add_argument("--no-recursive", action="store_true", help="Do not search recursively")
    ap.add_argument("--compare-v1", action="store_true", help="Also benchmark v1 (read-only)")
    ap.add_argument("--output", default="./results/bench_v2.json", help="Output JSON path")

    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    labels_path = Path(args.labels) if args.labels else None
    config_path = Path(args.config)

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    images = collect_images(data_dir, patterns, recursive=not args.no_recursive)
    if not images:
        print(f"No images found in {data_dir} with patterns={patterns}")
        sys.exit(1)

    labels_map = safe_read_json(labels_path)
    config = load_config(config_path)

    # v2 system
    sys_v2 = SafetyAssessmentSystemV2(config)
    bench_v2 = benchmark_system("v2", sys_v2, images, labels_map)

    results = {"v2": bench_v2}

    # Optional v1 comparison (import without modifications)
    if args.compare_v1:
        try:
            from src.core.safety_assessment_system import SafetyAssessmentSystem as SysV1
            sys_v1 = SysV1(config)
            bench_v1 = benchmark_system("v1", sys_v1, images, labels_map)
            results["v1"] = bench_v1
        except Exception as e:
            results["v1_error"] = str(e)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()

