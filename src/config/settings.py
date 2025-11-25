"""
Global settings and configuration for SafetyKnob system.

This module provides centralized configuration management with
support for environment variables and configuration files.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for vision models."""
    
    model_types: list = None
    device: str = "cuda"
    batch_size: int = 32
    cache_embeddings: bool = True
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["siglip", "clip", "dinov2", "evaclip"]


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    results_dir: str = "results"
    image_extensions: list = None
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    ensemble_method: str = "voting"
    cross_validation_folds: int = 5
    
    # Thresholds
    danger_threshold: float = 0.5
    confidence_threshold: float = 0.6


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    log_dir: str = "logs"
    console_logging: bool = True
    file_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class SafetyKnobConfig:
    """Main configuration class for SafetyKnob system."""
    
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "logging": asdict(self.logging)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SafetyKnobConfig':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            logging=LoggingConfig(**config_dict.get("logging", {}))
        )
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SafetyKnobConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class ConfigManager:
    """Manager for handling configuration with environment variable support."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or os.environ.get('SAFETYKNOB_CONFIG', 'config.json')
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> SafetyKnobConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_file):
            try:
                return SafetyKnobConfig.load(self.config_file)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_file}: {e}")
        
        return SafetyKnobConfig()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Model settings
        if 'SAFETYKNOB_DEVICE' in os.environ:
            self.config.model.device = os.environ['SAFETYKNOB_DEVICE']
        
        if 'SAFETYKNOB_BATCH_SIZE' in os.environ:
            self.config.model.batch_size = int(os.environ['SAFETYKNOB_BATCH_SIZE'])
        
        # Data settings
        if 'SAFETYKNOB_DATA_DIR' in os.environ:
            self.config.data.data_dir = os.environ['SAFETYKNOB_DATA_DIR']
        
        if 'SAFETYKNOB_CACHE_DIR' in os.environ:
            self.config.data.cache_dir = os.environ['SAFETYKNOB_CACHE_DIR']
        
        if 'SAFETYKNOB_RESULTS_DIR' in os.environ:
            self.config.data.results_dir = os.environ['SAFETYKNOB_RESULTS_DIR']
        
        # Training settings
        if 'SAFETYKNOB_TEST_SIZE' in os.environ:
            self.config.training.test_size = float(os.environ['SAFETYKNOB_TEST_SIZE'])
        
        if 'SAFETYKNOB_RANDOM_STATE' in os.environ:
            self.config.training.random_state = int(os.environ['SAFETYKNOB_RANDOM_STATE'])
        
        # Logging settings
        if 'SAFETYKNOB_LOG_LEVEL' in os.environ:
            self.config.logging.level = os.environ['SAFETYKNOB_LOG_LEVEL']
        
        if 'SAFETYKNOB_LOG_DIR' in os.environ:
            self.config.logging.log_dir = os.environ['SAFETYKNOB_LOG_DIR']
    
    def save(self, path: Optional[str] = None):
        """Save current configuration."""
        save_path = path or self.config_file
        self.config.save(save_path)
    
    def get(self) -> SafetyKnobConfig:
        """Get current configuration."""
        return self.config
    
    def update(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like "model.device"
                parts = key.split('.')
                obj = self.config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self.config, key, value)


# Global configuration instance
_config_manager = None


def get_config() -> SafetyKnobConfig:
    """Get global configuration instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager.get()


def update_config(**kwargs):
    """Update global configuration."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    _config_manager.update(**kwargs)


def save_config(path: Optional[str] = None):
    """Save global configuration."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    _config_manager.save(path)


class SystemConfig:
    """System configuration compatible with main.py and config.json structure."""
    
    def __init__(self):
        self.models = []
        self.assessment_method = "ensemble"
        self.ensemble_strategy = "weighted_vote"
        self.safety = {
            "dimensions": {},
            "safety_threshold": 0.5,
            "confidence_threshold": 0.7
        }
        self.training = TrainingConfig()
        self.checkpoint_dir = "./checkpoints"
        self.api_host = "0.0.0.0"
        self.api_port = 8000
        self.api_workers = 1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create SystemConfig from dictionary (config.json format)."""
        config = cls()
        
        # Load models
        config.models = config_dict.get("models", [])
        
        # Load assessment settings
        config.assessment_method = config_dict.get("assessment_method", "ensemble")
        config.ensemble_strategy = config_dict.get("ensemble_strategy", "weighted_vote")
        
        # Load safety settings
        config.safety = config_dict.get("safety", config.safety)
        
        # Load training settings
        training_dict = config_dict.get("training", {})
        config.training = TrainingConfig(
            test_size=training_dict.get("test_size", 0.2),
            validation_size=training_dict.get("validation_size", 0.1),
            random_state=training_dict.get("random_state", 42),
            ensemble_method=training_dict.get("ensemble_method", "voting"),
            cross_validation_folds=training_dict.get("cross_validation_folds", 5),
            danger_threshold=training_dict.get("danger_threshold", 0.5),
            confidence_threshold=training_dict.get("confidence_threshold", 0.6)
        )
        
        # Additional training parameters
        config.training.batch_size = training_dict.get("batch_size", 32)
        config.training.learning_rate = training_dict.get("learning_rate", 0.001)
        config.training.epochs = training_dict.get("epochs", 20)
        config.training.optimizer = training_dict.get("optimizer", "adam")
        config.training.scheduler = training_dict.get("scheduler", "cosine")
        config.training.early_stopping_patience = training_dict.get("early_stopping_patience", 5)
        config.training.weight_decay = training_dict.get("weight_decay", 0.0001)
        
        # Load other settings
        config.checkpoint_dir = config_dict.get("checkpoint_dir", "./checkpoints")
        config.api_host = config_dict.get("api_host", "0.0.0.0")
        config.api_port = config_dict.get("api_port", 8000)
        config.api_workers = config_dict.get("api_workers", 1)
        
        return config