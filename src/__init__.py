"""
SafetyKnob Safety Classification System.

A multi-model vision-based safety classification system for detecting
dangerous situations in images using ensemble methods.
"""

__version__ = "2.0.0"
__author__ = "SafetyKnob Team"
__license__ = "MIT"

# Configure logging on import
from .utils.logger import configure_root_logger
configure_root_logger()

# Main exports
from .core import (
    SafetyClassifier,
    create_embedder
)

from .analysis import (
    SingleModelAnalyzer,
    MultiModelAnalyzer,
    MetricsCalculator
)

from .config import (
    get_config,
    update_config,
    get_path,
    join_path
)

__all__ = [
    # Version info
    '__version__',
    
    # Core functionality
    'SafetyClassifier',
    'create_embedder',
    
    # Analysis
    'SingleModelAnalyzer',
    'MultiModelAnalyzer',
    'MetricsCalculator',
    
    # Configuration
    'get_config',
    'update_config',
    'get_path',
    'join_path'
]