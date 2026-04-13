"""
Path configuration and management.

This module provides centralized path management for the SafetyKnob system,
ensuring consistent file and directory access across all modules.
"""

import os
from pathlib import Path
from typing import Optional, Union


class PathManager:
    """Manager for all system paths."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize path manager.
        
        Args:
            base_dir: Base directory for the project
        """
        if base_dir is None:
            # Try to find project root
            current = Path(__file__).resolve()
            # Go up to find project root (contains src directory)
            while current.parent != current:
                if (current / "src").exists():
                    base_dir = str(current)
                    break
                current = current.parent
            else:
                # Fallback to current directory
                base_dir = os.getcwd()
        
        self.base_dir = Path(base_dir).resolve()
        
        # Define directory structure
        self._dirs = {
            # Source code
            "src": self.base_dir / "src",
            "core": self.base_dir / "src" / "core",
            "analysis": self.base_dir / "src" / "analysis",
            "utils": self.base_dir / "src" / "utils",
            "api": self.base_dir / "src" / "api",
            "config": self.base_dir / "src" / "config",
            
            # Data directories
            "data": self.base_dir / "data",
            "raw_data": self.base_dir / "data" / "raw",
            "processed_data": self.base_dir / "data" / "processed",
            "cache": self.base_dir / "data" / "cache",
            
            # Model directories
            "models": self.base_dir / "models",
            "checkpoints": self.base_dir / "models" / "checkpoints",
            
            # Results directories
            "results": self.base_dir / "results",
            "plots": self.base_dir / "results" / "plots",
            "reports": self.base_dir / "results" / "reports",
            "archives": self.base_dir / "results" / "archives",
            
            # Other directories
            "logs": self.base_dir / "logs",
            "scripts": self.base_dir / "scripts",
            "tests": self.base_dir / "tests",
            "docs": self.base_dir / "docs",
            "notebooks": self.base_dir / "notebooks"
        }
        
        # Create essential directories
        self._create_directories()
    
    def _create_directories(self):
        """Create essential directories if they don't exist."""
        essential_dirs = [
            "data", "cache", "models", "results", "logs"
        ]
        
        for dir_name in essential_dirs:
            if dir_name in self._dirs:
                self._dirs[dir_name].mkdir(parents=True, exist_ok=True)
    
    def get(self, name: str) -> Path:
        """
        Get path for a named directory.
        
        Args:
            name: Directory name
            
        Returns:
            Path object
        """
        if name not in self._dirs:
            raise ValueError(f"Unknown directory: {name}")
        
        return self._dirs[name]
    
    def get_str(self, name: str) -> str:
        """Get path as string."""
        return str(self.get(name))
    
    def join(self, name: str, *parts) -> Path:
        """
        Join path parts to a named directory.
        
        Args:
            name: Directory name
            *parts: Path parts to join
            
        Returns:
            Joined path
        """
        base = self.get(name)
        return base.joinpath(*parts)
    
    def join_str(self, name: str, *parts) -> str:
        """Join path parts and return as string."""
        return str(self.join(name, *parts))
    
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_data_path(self, category: str, subset: Optional[str] = None) -> Path:
        """
        Get path for data category.
        
        Args:
            category: Data category (e.g., "danger", "safe")
            subset: Data subset (e.g., "train", "test")
            
        Returns:
            Data path
        """
        if subset:
            return self.join("processed_data", subset, category)
        else:
            return self.join("raw_data", category)
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """
        Get path for model files.
        
        Args:
            model_name: Model name
            version: Model version
            
        Returns:
            Model path
        """
        if version:
            return self.join("models", model_name, version)
        else:
            return self.join("models", model_name)
    
    def get_result_path(
        self,
        experiment: str,
        timestamp: Optional[str] = None,
        create: bool = True
    ) -> Path:
        """
        Get path for experiment results.
        
        Args:
            experiment: Experiment name
            timestamp: Timestamp string
            create: Whether to create directory
            
        Returns:
            Result path
        """
        if timestamp:
            path = self.join("results", experiment, timestamp)
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.join("results", experiment, timestamp)
        
        if create:
            self.ensure_dir(path)
        
        return path
    
    def get_cache_path(self, cache_type: str, identifier: str) -> Path:
        """
        Get path for cache file.
        
        Args:
            cache_type: Type of cache
            identifier: Cache identifier
            
        Returns:
            Cache file path
        """
        cache_dir = self.join("cache", cache_type)
        self.ensure_dir(cache_dir)
        
        # Sanitize identifier for filename
        safe_id = identifier.replace("/", "_").replace("\\", "_")
        return cache_dir / f"{safe_id}.pkl"
    
    def list_files(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> list:
        """
        List files in directory.
        
        Args:
            directory: Directory name
            pattern: File pattern
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        dir_path = self.get(directory)
        
        if recursive:
            return sorted(dir_path.rglob(pattern))
        else:
            return sorted(dir_path.glob(pattern))
    
    def clean_directory(self, directory: str, pattern: str = "*"):
        """
        Clean files from directory.
        
        Args:
            directory: Directory name
            pattern: File pattern to remove
        """
        dir_path = self.get(directory)
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
    
    def get_latest_file(self, directory: str, pattern: str = "*") -> Optional[Path]:
        """
        Get the most recently modified file.
        
        Args:
            directory: Directory name
            pattern: File pattern
            
        Returns:
            Path to latest file or None
        """
        files = self.list_files(directory, pattern)
        
        if not files:
            return None
        
        return max(files, key=lambda p: p.stat().st_mtime)
    
    def __str__(self) -> str:
        """String representation."""
        return f"PathManager(base_dir={self.base_dir})"
    
    def summary(self) -> str:
        """Get summary of all paths."""
        lines = ["Path Configuration:"]
        lines.append(f"Base Directory: {self.base_dir}")
        lines.append("\nDirectories:")
        
        for name, path in sorted(self._dirs.items()):
            exists = "✓" if path.exists() else "✗"
            lines.append(f"  {exists} {name}: {path}")
        
        return "\n".join(lines)


# Global path manager instance
_path_manager = None


def get_path_manager(base_dir: Optional[str] = None) -> PathManager:
    """Get global path manager instance."""
    global _path_manager
    
    if _path_manager is None:
        _path_manager = PathManager(base_dir)
    
    return _path_manager


# Convenience functions
def get_path(name: str) -> Path:
    """Get path for named directory."""
    return get_path_manager().get(name)


def join_path(name: str, *parts) -> Path:
    """Join path parts to named directory."""
    return get_path_manager().join(name, *parts)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    return get_path_manager().ensure_dir(path)