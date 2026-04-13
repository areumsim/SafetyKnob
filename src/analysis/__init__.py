"""
Analysis module for SafetyKnob safety classification system.

This module provides tools for analyzing and comparing model performance,
including single model analysis, multi-model comparison, and comprehensive
metrics calculation.
"""

# Define classes even if imports fail
class SingleModelAnalyzer:
    """Placeholder for single model analyzer"""
    pass

class MultiModelAnalyzer:
    """Placeholder for multi model analyzer"""
    pass

class MetricsCalculator:
    """Placeholder for metrics calculator"""
    pass

# Try to import actual implementations
try:
    from .model_comparison import ModelPerformanceAnalyzer
except ImportError:
    ModelPerformanceAnalyzer = None

__all__ = [
    'SingleModelAnalyzer',
    'MultiModelAnalyzer',
    'MetricsCalculator'
]

if ModelPerformanceAnalyzer:
    __all__.append('ModelPerformanceAnalyzer')