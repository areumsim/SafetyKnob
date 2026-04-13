"""
FastAPI server for Industrial Image Safety Assessment
"""

import io
import time
import uuid
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from ..core import SafetyAssessmentSystem
from ..config.settings import SystemConfig

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_BATCH_SIZE = 50


def create_app(config: Optional[SystemConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Industrial Image Safety Assessment API",
        description="API for assessing safety in industrial images using AI",
        version="1.0.0"
    )
    
    # Configure CORS (restrict in production)
    # For production, set specific origins instead of ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Replace with specific origins in production
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Initialize system
    config = config or SystemConfig()
    system = SafetyAssessmentSystem(config)
    app.state.system = system
    app.state.config = config
    
    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Industrial Image Safety Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "assess": "/api/v1/assess",
            "info": "/api/v1/info"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "industrial-safety-assessment",
        "models_loaded": len(app.state.config.models),
        "assessment_method": app.state.config.assessment_method
    }


@app.get("/api/v1/info")
async def system_info():
    """Get system information"""
    config = app.state.config
    return {
        "system_name": "Industrial Image Safety Assessment",
        "version": "1.0.0",
        "models": [
            {
                "name": model.name,
                "type": model.model_type,
                "embedding_dim": model.embedding_dim
            }
            for model in config.models
        ],
        "safety_dimensions": config.safety.dimensions,
        "assessment_method": config.assessment_method,
        "ensemble_strategy": config.ensemble_strategy
    }


@app.post("/api/v1/assess")
async def assess_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Assess safety of an uploaded image

    Args:
        file: Image file to assess

    Returns:
        Safety assessment results
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": "File must be an image",
            "allowed_types": ["image/jpeg", "image/png", "image/bmp"]
        })

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": f"Invalid file extension: {file_ext}",
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        })

    try:
        # Read file contents
        contents = await file.read()

        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail={
                "success": False,
                "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            })

        # Open and validate image
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()  # Verify it's a valid image
            image = Image.open(io.BytesIO(contents))  # Reopen after verify
        except Exception as e:
            raise HTTPException(status_code=400, detail={
                "success": False,
                "error": f"Invalid or corrupted image: {str(e)}"
            })

        # Use secure temporary file with UUID
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
            image.save(temp_path)

        try:
            # Perform assessment
            start_time = time.time()
            result = app.state.system.assess_image(str(temp_path))
            processing_time = time.time() - start_time

            # Update processing time
            result.processing_time = processing_time

            return {
                "success": True,
                "result": result.to_dict()
            }

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": f"Assessment failed: {str(e)}",
            "type": type(e).__name__
        })


@app.post("/api/v1/assess/batch")
async def assess_batch(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Assess safety of multiple images

    Args:
        files: List of image files to assess

    Returns:
        Batch assessment results
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": f"Maximum {MAX_BATCH_SIZE} images per batch",
            "received": len(files)
        })

    results = []
    errors = []
    temp_files = []

    try:
        for i, file in enumerate(files):
            temp_path = None
            try:
                # Validate file type
                if not file.content_type.startswith("image/"):
                    errors.append({
                        "index": i,
                        "filename": file.filename,
                        "error": "File must be an image"
                    })
                    continue

                # Validate file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    errors.append({
                        "index": i,
                        "filename": file.filename,
                        "error": f"Invalid extension: {file_ext}"
                    })
                    continue

                # Read and validate size
                contents = await file.read()
                if len(contents) > MAX_FILE_SIZE:
                    errors.append({
                        "index": i,
                        "filename": file.filename,
                        "error": f"File too large (max {MAX_FILE_SIZE / (1024*1024):.1f}MB)"
                    })
                    continue

                # Open and validate image
                try:
                    image = Image.open(io.BytesIO(contents))
                    image.verify()
                    image = Image.open(io.BytesIO(contents))
                except Exception as e:
                    errors.append({
                        "index": i,
                        "filename": file.filename,
                        "error": f"Invalid image: {str(e)}"
                    })
                    continue

                # Use secure temporary file
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                    temp_path = Path(tmp_file.name)
                    temp_files.append(temp_path)
                    image.save(temp_path)

                # Perform assessment
                result = app.state.system.assess_image(str(temp_path))
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "result": result.to_dict()
                })

            except Exception as e:
                errors.append({
                    "index": i,
                    "filename": file.filename,
                    "error": str(e),
                    "type": type(e).__name__
                })

        return {
            "success": len(errors) == 0,
            "total": len(files),
            "processed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors if errors else None
        }

    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "type": type(exc).__name__
        }
    )


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )