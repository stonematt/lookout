"""
Memory utilities for monitoring and cleanup across the Lookout application.

Centralized memory monitoring to avoid import duplication and ensure
consistent memory tracking patterns.
"""

import gc
import sys
from typing import Dict, Any, Optional

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Constants for memory calculations
BYTES_TO_MB = 1024 / 1024
MEMORY_UNAVAILABLE = 0.0


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / BYTES_TO_MB
    except ImportError:
        return MEMORY_UNAVAILABLE


def log_memory_usage(context: str, start_memory: Optional[float] = None) -> float:
    """
    Log memory usage with optional delta calculation.

    :param context: Description of the operation
    :param start_memory: Starting memory for delta calculation
    :return: Current memory usage in MB
    """
    current_memory = get_memory_usage()

    if start_memory is not None:
        delta = current_memory - start_memory
        logger.debug(f"{context}: {current_memory:.1f}MB ({delta:+.1f}MB)")
    else:
        logger.debug(f"{context}: {current_memory:.1f}MB")

    return current_memory


def force_garbage_collection() -> int:
    """
    Force multiple garbage collection cycles and return object count.

    :return: Number of objects after GC
    """
    gc.collect()
    for _ in range(2):  # Additional cycles for circular references
        gc.collect()
    return len(gc.get_objects())


def cleanup_cache_functions(*cache_funcs) -> None:
    """
    Safely clear cache functions if available.

    :param cache_funcs: Variable number of cache functions to clear
    """
    for cache_func in cache_funcs:
        try:
            if hasattr(cache_func, "clear"):
                cache_func.clear()
        except Exception as e:
            logger.warning(f"Cache clearing failed: {e}")


def get_object_counts() -> Dict[str, int]:
    """
    Get counts of specific object types for memory analysis.

    :return: Dictionary with object type counts
    """
    gc.collect()
    import pandas as pd

    all_objects = gc.get_objects()
    return {
        "total_objects": len(all_objects),
        "dataframes": len(
            [obj for obj in all_objects if isinstance(obj, pd.DataFrame)]
        ),
        "plotly_objects": len(
            [obj for obj in all_objects if "plotly" in str(type(obj)).lower()]
        ),
    }


def get_df_memory_usage(df) -> float:
    """
    Get accurate DataFrame memory usage in MB.

    :param df: pandas DataFrame
    :return: Memory usage in MB
    """
    try:
        return df.memory_usage(deep=True).sum() / BYTES_TO_MB
    except:
        return sys.getsizeof(df) / BYTES_TO_MB


def get_object_memory_usage(obj) -> float:
    """
    Get object memory usage in MB.

    :param obj: Any object
    :return: Memory usage in MB
    """
    return sys.getsizeof(obj) / BYTES_TO_MB
