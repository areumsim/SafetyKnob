"""
Utility modules for SafetyKnob safety classification system.

This package provides various utility functions for data loading,
visualization, caching, and logging.
"""

# Import core utilities
try:
    from .data_utils import ImageDataset, load_image, prepare_dataloader
except ImportError:
    # Fallback for legacy imports
    from .data_loader import (
        load_image_dataset,
        organize_by_scenario,
        split_dataset,
        load_paired_dataset,
        create_balanced_dataset,
        save_dataset_info
    )

from .visualization import (
    setup_plotting_style,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_embedding_space,
    plot_model_comparison,
    plot_threshold_analysis,
    create_performance_report
)

from .cache_manager import (
    CacheManager,
    get_cache_manager
)

from .logger import (
    setup_logging,
    get_logger,
    configure_root_logger,
    LoggerContext,
    log_function_call,
    log_execution_time
)

__all__ = [
    # Data loader
    'load_image_dataset',
    'organize_by_scenario',
    'split_dataset',
    'load_paired_dataset',
    'create_balanced_dataset',
    'save_dataset_info',
    
    # Visualization
    'setup_plotting_style',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_embedding_space',
    'plot_model_comparison',
    'plot_threshold_analysis',
    'create_performance_report',
    
    # Cache manager
    'CacheManager',
    'get_cache_manager',
    
    # Logger
    'setup_logging',
    'get_logger',
    'configure_root_logger',
    'LoggerContext',
    'log_function_call',
    'log_execution_time'
]