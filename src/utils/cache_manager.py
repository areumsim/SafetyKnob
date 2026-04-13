"""
Cache management utilities.

This module provides functions for managing embedding caches
and other cached data to improve performance.
"""

import os
import pickle
import json
import hashlib
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import shutil


logger = logging.getLogger(__name__)


class CacheManager:
    """Manager for handling various types of caches."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base directory for all caches
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_cache_path(self, cache_type: str, identifier: str) -> str:
        """
        Get cache file path.
        
        Args:
            cache_type: Type of cache (e.g., "embeddings", "models")
            identifier: Unique identifier for this cache
            
        Returns:
            Full path to cache file
        """
        cache_subdir = os.path.join(self.cache_dir, cache_type)
        os.makedirs(cache_subdir, exist_ok=True)
        
        # Create safe filename
        safe_id = hashlib.md5(identifier.encode()).hexdigest()[:16]
        filename = f"{identifier.replace('/', '_')}_{safe_id}.pkl"
        
        return os.path.join(cache_subdir, filename)
    
    def save(
        self,
        data: Any,
        cache_type: str,
        identifier: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            cache_type: Type of cache
            identifier: Unique identifier
            metadata: Optional metadata about the cache
        """
        cache_path = self.get_cache_path(cache_type, identifier)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            cache_key = f"{cache_type}/{identifier}"
            self.metadata[cache_key] = {
                "path": cache_path,
                "timestamp": datetime.now().isoformat(),
                "size": os.path.getsize(cache_path),
                "metadata": metadata or {}
            }
            self._save_metadata()
            
            logger.info(f"Cached {cache_type}/{identifier} ({os.path.getsize(cache_path) / 1024 / 1024:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise
    
    def load(
        self,
        cache_type: str,
        identifier: str,
        max_age: Optional[timedelta] = None
    ) -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            cache_type: Type of cache
            identifier: Unique identifier
            max_age: Maximum age of cache to consider valid
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_key = f"{cache_type}/{identifier}"
        
        if cache_key not in self.metadata:
            return None
        
        cache_info = self.metadata[cache_key]
        cache_path = cache_info["path"]
        
        if not os.path.exists(cache_path):
            # Cache file missing, clean up metadata
            del self.metadata[cache_key]
            self._save_metadata()
            return None
        
        # Check age if specified
        if max_age:
            timestamp = datetime.fromisoformat(cache_info["timestamp"])
            if datetime.now() - timestamp > max_age:
                logger.info(f"Cache {cache_key} expired")
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded cache {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def exists(self, cache_type: str, identifier: str) -> bool:
        """Check if cache exists."""
        cache_key = f"{cache_type}/{identifier}"
        if cache_key not in self.metadata:
            return False
        
        cache_path = self.metadata[cache_key]["path"]
        return os.path.exists(cache_path)
    
    def delete(self, cache_type: str, identifier: str):
        """Delete specific cache."""
        cache_key = f"{cache_type}/{identifier}"
        
        if cache_key in self.metadata:
            cache_path = self.metadata[cache_key]["path"]
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Deleted cache {cache_key}")
            
            del self.metadata[cache_key]
            self._save_metadata()
    
    def clear_type(self, cache_type: str):
        """Clear all caches of a specific type."""
        cache_subdir = os.path.join(self.cache_dir, cache_type)
        
        if os.path.exists(cache_subdir):
            shutil.rmtree(cache_subdir)
            logger.info(f"Cleared all {cache_type} caches")
        
        # Update metadata
        keys_to_remove = [
            key for key in self.metadata
            if key.startswith(f"{cache_type}/")
        ]
        
        for key in keys_to_remove:
            del self.metadata[key]
        
        self._save_metadata()
    
    def clear_all(self):
        """Clear all caches."""
        if os.path.exists(self.cache_dir):
            # Keep the directory but remove contents
            for item in os.listdir(self.cache_dir):
                item_path = os.path.join(self.cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        self.metadata = {}
        self._save_metadata()
        logger.info("Cleared all caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_size": 0,
            "num_caches": len(self.metadata),
            "by_type": {},
            "oldest": None,
            "newest": None
        }
        
        oldest_time = None
        newest_time = None
        
        for cache_key, info in self.metadata.items():
            cache_type = cache_key.split('/')[0]
            
            # Size stats
            size = info.get("size", 0)
            stats["total_size"] += size
            
            if cache_type not in stats["by_type"]:
                stats["by_type"][cache_type] = {
                    "count": 0,
                    "size": 0
                }
            
            stats["by_type"][cache_type]["count"] += 1
            stats["by_type"][cache_type]["size"] += size
            
            # Time stats
            timestamp = datetime.fromisoformat(info["timestamp"])
            if oldest_time is None or timestamp < oldest_time:
                oldest_time = timestamp
                stats["oldest"] = cache_key
            
            if newest_time is None or timestamp > newest_time:
                newest_time = timestamp
                stats["newest"] = cache_key
        
        # Convert size to MB
        stats["total_size_mb"] = stats["total_size"] / 1024 / 1024
        for cache_type in stats["by_type"]:
            size_bytes = stats["by_type"][cache_type]["size"]
            stats["by_type"][cache_type]["size_mb"] = size_bytes / 1024 / 1024
        
        return stats
    
    def cleanup_old_caches(self, max_age: timedelta):
        """Remove caches older than specified age."""
        current_time = datetime.now()
        keys_to_remove = []
        
        for cache_key, info in self.metadata.items():
            timestamp = datetime.fromisoformat(info["timestamp"])
            
            if current_time - timestamp > max_age:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            cache_type, identifier = key.split('/', 1)
            self.delete(cache_type, identifier)
        
        logger.info(f"Removed {len(keys_to_remove)} old caches")
    
    def get_or_compute(
        self,
        cache_type: str,
        identifier: str,
        compute_fn,
        max_age: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get from cache or compute if not available.
        
        Args:
            cache_type: Type of cache
            identifier: Unique identifier
            compute_fn: Function to compute data if not cached
            max_age: Maximum age of cache to consider valid
            metadata: Optional metadata about the cache
            
        Returns:
            Cached or computed data
        """
        # Try to load from cache
        data = self.load(cache_type, identifier, max_age)
        
        if data is not None:
            return data
        
        # Compute and cache
        logger.info(f"Computing {cache_type}/{identifier}...")
        data = compute_fn()
        
        self.save(data, cache_type, identifier, metadata)
        
        return data


# Global cache manager instance
_cache_manager = None


def get_cache_manager(cache_dir: str = "data/cache") -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir)
    
    return _cache_manager