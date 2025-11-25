"""
API module for SafetyKnob safety classification system.

This module provides high-level API interfaces for inference
and integration with other systems.
"""

# Import server functions if available
try:
    from .server import create_app, run_server
    __all__ = ['create_app', 'run_server']
except ImportError:
    # Fallback to legacy imports
    try:
        from .inference import SafetyInferenceAPI, quick_predict
        __all__ = ['SafetyInferenceAPI', 'quick_predict']
    except ImportError:
        __all__ = []