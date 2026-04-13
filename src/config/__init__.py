"""
Configuration module for SafetyKnob safety classification system.

This module provides centralized configuration management including
settings, paths, and environment variable support.
"""

# Import new configuration
try:
    from .settings import SystemConfig, DEFAULT_CONFIG
    __all__ = ['SystemConfig', 'DEFAULT_CONFIG']
except ImportError:
    # Fallback to legacy imports
    try:
        from .settings import (
            ModelConfig,
            DataConfig,
            TrainingConfig,
            LoggingConfig,
            SafetyKnobConfig,
            ConfigManager,
            get_config,
            update_config,
            save_config
        )
        
        from .paths import (
            PathManager,
            get_path_manager,
            get_path,
            join_path,
            ensure_dir
        )
        
        __all__ = [
            # Settings
            'ModelConfig',
            'DataConfig',
            'TrainingConfig',
            'LoggingConfig',
            'SafetyKnobConfig',
            'ConfigManager',
            'get_config',
            'update_config',
            'save_config',
            
            # Paths
            'PathManager',
            'get_path_manager',
            'get_path',
            'join_path',
            'ensure_dir'
        ]
    except ImportError:
        __all__ = []