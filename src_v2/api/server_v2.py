"""
FastAPI server (v2) for Industrial Image Safety Assessment
Uses SafetyAssessmentSystemV2 and addresses API/doc consistency and temp-file safety.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from ..core import SafetyAssessmentSystemV2
from ...config.settings import SystemConfig


MAX_FILE_SIZE_MB = 10
ALLOWED_MIME = {"image/jpeg", "image/png", "image/bmp"}


def _validate_upload(file: UploadFile, contents: bytes):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_SIZE_MB}MB)")


def create_app_v2(config: Optional[SystemConfig] = None) -> FastAPI:
    app = FastAPI(
        title="Industrial Image Safety Assessment API (v2)",
        description="API for assessing safety in industrial images using AI (v2)",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    cfg = config or SystemConfig()
    system = SafetyAssessmentSystemV2(cfg)
    app.state.system = system
    app.state.config = cfg

    @app.get("/")
    async def root():
        return {
            "name": "Industrial Image Safety Assessment API (v2)",
            "version": "2.0.0",
            "endpoints": {
                "health": "/api/v1/health",
                "assess": "/api/v1/assess",
                "assess_batch": "/api/v1/assess/batch",
                "info": "/api/v1/info",
                "models": "/api/v1/models",
            },
        }

    @app.get("/api/v1/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "industrial-safety-assessment-v2",
            "models_loaded": len(app.state.config.models),
            "assessment_method": getattr(app.state.config, "assessment_method", "ensemble"),
        }

    @app.get("/api/v1/info")
    async def system_info():
        cfg = app.state.config
        return {
            "system_name": "Industrial Image Safety Assessment (v2)",
            "version": "2.0.0",
            "models": [
                {
                    "name": m.get("name") if isinstance(m, dict) else m.name,
                    "type": m.get("model_type") if isinstance(m, dict) else m.model_type,
                    "embedding_dim": m.get("embedding_dim", 0) if isinstance(m, dict) else getattr(m, "embedding_dim", 0),
                    "checkpoint": m.get("checkpoint") if isinstance(m, dict) else getattr(m, "checkpoint", None),
                    "device": m.get("device") if isinstance(m, dict) else getattr(m, "device", None),
                }
                for m in cfg.models
            ],
            "safety_dimensions": cfg.safety["dimensions"] if isinstance(cfg.safety, dict) else cfg.safety.dimensions,
            "assessment_method": getattr(cfg, "assessment_method", "ensemble"),
            "ensemble_strategy": getattr(cfg, "ensemble_strategy", "weighted_vote"),
        }

    @app.get("/api/v1/models")
    async def list_models():
        cfg = app.state.config
        return {
            "models": [
                {
                    "name": m.get("name") if isinstance(m, dict) else m.name,
                    "type": m.get("model_type") if isinstance(m, dict) else m.model_type,
                    "checkpoint": m.get("checkpoint") if isinstance(m, dict) else getattr(m, "checkpoint", None),
                    "embedding_dim": m.get("embedding_dim", 0) if isinstance(m, dict) else getattr(m, "embedding_dim", 0),
                    "status": "loaded",
                }
                for m in cfg.models
            ],
            "ensemble_enabled": True,
            "ensemble_strategy": getattr(cfg, "ensemble_strategy", "weighted_vote"),
        }

    @app.post("/api/v1/assess")
    async def assess_image(file: UploadFile = File(...)) -> Dict[str, Any]:
        contents = await file.read()
        _validate_upload(file, contents)
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        # Use a safe temp file name; delete after use
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".jpg")
        try:
            image.save(tmp.name)
            t0 = time.time()
            res = app.state.system.assess_image(tmp.name)
            t1 = time.time()
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        res.processing_time = t1 - t0
        return {"success": True, "result": res.to_dict()}

    @app.post("/api/v1/assess/batch")
    async def assess_batch(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 images per batch")
        results, errors = [], []
        for idx, f in enumerate(files):
            try:
                contents = await f.read()
                _validate_upload(f, contents)
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix or ".jpg")
                try:
                    image.save(tmp.name)
                    res = app.state.system.assess_image(tmp.name)
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
                results.append({"index": idx, "filename": f.filename, "result": res.to_dict()})
            except HTTPException as he:
                errors.append({"index": idx, "filename": f.filename, "error": he.detail})
            except Exception as e:
                errors.append({"index": idx, "filename": f.filename, "error": str(e)})
        return {
            "success": len(errors) == 0,
            "total": len(files),
            "processed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc), "type": type(exc).__name__},
        )

    return app


def run_server_v2(app: FastAPI, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")

