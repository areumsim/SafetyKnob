#!/usr/bin/env python3
"""
Experimental v2 entrypoint for Industrial Image Safety Assessment System

This mirrors main.py but routes core operations to v2 modules so you can
benchmark/compare without touching the original pipeline.
"""

import argparse
import json
import logging
from pathlib import Path
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src_v2.core import SafetyAssessmentSystemV2
from src_v2.api import create_app_v2, run_server_v2
from src.config.settings import SystemConfig

logger = logging.getLogger(__name__)


def configure_logging(debug=False, verbose=False):
    env_log_level = os.environ.get("SAFETYKNOB_LOG_LEVEL", "").upper()
    if debug or os.environ.get("SAFETYKNOB_DEBUG", ""):
        lvl = logging.DEBUG
    elif verbose:
        lvl = logging.INFO
    elif env_log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        lvl = getattr(logging, env_log_level)
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Industrial Image Safety Assessment System (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command", help="Command")

    assess = sub.add_parser("assess", help="Assess safety of image or directory (v2)")
    assess.add_argument("path", type=str)
    assess.add_argument("--config", type=str, default="config.json")
    assess.add_argument("--recursive", action="store_true")
    assess.add_argument("--output", type=str)
    assess.add_argument("--pattern", type=str, default="*.jpg")

    evaluate = sub.add_parser("evaluate", help="Evaluate on a dataset (v2)")
    evaluate.add_argument("--data-dir", type=str, required=True)
    evaluate.add_argument("--labels", type=str)
    evaluate.add_argument("--config", type=str, default="config.json")
    evaluate.add_argument("--output", type=str, default="evaluation_results_v2.json")

    train = sub.add_parser("train", help="Train the system (v2)")
    train.add_argument("--data-dir", type=str, required=True)
    train.add_argument("--labels", type=str)
    train.add_argument("--config", type=str, default="config.json")
    train.add_argument("--epochs", type=int)

    serve = sub.add_parser("serve", help="Start API server (v2)")
    serve.add_argument("--host", type=str, default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--config", type=str, default="config.json")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    configure_logging(args.debug, args.verbose)

    # Load config
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = json.load(f)
        config = SystemConfig.from_dict(cfg)
    else:
        logger.warning("Config file not found; using defaults")
        config = SystemConfig()

    system = SafetyAssessmentSystemV2(config)

    if args.command == "assess":
        p = Path(args.path)
        if p.is_file():
            res = system.assess_image(str(p))
            print("\n" + "=" * 60)
            print("Safety Assessment (v2)")
            print("=" * 60)
            print(f"Image: {res.image_path}")
            print(f"Overall Safety: {'SAFE' if res.is_safe else 'UNSAFE'}")
            print(f"Safety Score: {res.overall_safety_score:.2%}")
            print(f"Confidence: {res.confidence:.2%}")
            print(f"Method: {res.method_used}")
            print(f"Model: {res.model_name}")
            print(f"Time: {res.processing_time:.3f}s")
        elif p.is_dir():
            # reuse test_batch for batch I/O and reporting
            from test_batch import test_batch_images

            images = []
            patterns = args.pattern.split(",")
            if args.recursive:
                for pattern in patterns:
                    images.extend(str(x) for x in p.rglob(pattern))
            else:
                for pattern in patterns:
                    images.extend(str(x) for x in p.glob(pattern))
            if not images:
                print(f"No images found in {p} (pattern={args.pattern})")
                return
            out = args.output or f"batch_results_v2_{p.name}.json"
            test_batch_images(images, out)
        else:
            print(f"Path not found: {args.path}")

    elif args.command == "train":
        if args.epochs:
            config.training.epochs = args.epochs
        from src.utils import ImageDataset

        ds = ImageDataset(args.data_dir, Path(args.labels) if args.labels else None)
        system.train(ds, config.training)
        system.save_models()
        print("Training completed (v2)")

    elif args.command == "evaluate":
        from src.utils import ImageDataset

        # Try to load any saved models
        ckpt_dir = Path(config.checkpoint_dir)
        if ckpt_dir.exists():
            system.load_models()

        ds = ImageDataset(args.data_dir, Path(args.labels) if args.labels else None)
        results = system.evaluate_dataset(ds)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation results to {args.output}")

    elif args.command == "serve":
        # Use v2 FastAPI server
        app = create_app_v2(config)
        run_server_v2(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
