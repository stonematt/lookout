"""
Memory utilities for monitoring and cleanup across the Lookout application.

Centralized memory monitoring to avoid import duplication and ensure
consistent memory tracking patterns. Figures are in MB of 1024 * 1024 bytes.

Offers:

- `peak_rss_mb()`: the process's peak RSS so far (its high-water mark), from
  the standard library, so it needs no psutil; it only ever rises.
- `track_step(name)`: a context manager that measures one step: its
  duration, the peak RSS after it, and the tracemalloc peak during it and
  the traced memory still held after it, filling a `StepMemory`.
- `get_memory_usage()` / `log_memory_usage()`: the current RSS, through
  psutil when it is installed (it is not a dependency; 0.0 without it).
- `force_garbage_collection()`, `cleanup_cache_functions()`,
  `get_object_counts()`, `get_df_memory_usage()`, `get_object_memory_usage()`.
"""

import gc
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional

from lookout.utils.log_util import app_logger

if TYPE_CHECKING:
    import pandas as pd

logger = app_logger(__name__)

# Constants for memory calculations
BYTES_TO_MB = 1024 * 1024
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


def cleanup_cache_functions(*cache_funcs: object) -> None:
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


def get_df_memory_usage(df: "pd.DataFrame") -> float:
    """
    Get accurate DataFrame memory usage in MB.

    :param df: pandas DataFrame
    :return: Memory usage in MB
    """
    try:
        return df.memory_usage(deep=True).sum() / BYTES_TO_MB
    except Exception:
        return sys.getsizeof(df) / BYTES_TO_MB


def get_object_memory_usage(obj: object) -> float:
    """
    Get object memory usage in MB.

    :param obj: Any object
    :return: Memory usage in MB
    """
    return sys.getsizeof(obj) / BYTES_TO_MB


def peak_rss_mb() -> float:
    """
    The process's peak RSS so far, its high-water mark, in MB.

    Read from `resource.getrusage(RUSAGE_SELF).ru_maxrss`, which Linux gives
    in kB and macOS in bytes. The mark never falls, so after a run of steps
    the step where it jumps is the one that set it.

    :return: The peak in MB; `MEMORY_UNAVAILABLE` where `resource` is absent.
    """
    try:
        import resource
    except ImportError:
        return MEMORY_UNAVAILABLE
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1 if sys.platform == "darwin" else 1024
    return peak * unit / BYTES_TO_MB


@dataclass
class StepMemory:
    """
    What one step cost, as `track_step` measured it.

    :param step: The step's name.
    :param seconds: Its wall-clock duration.
    :param rss_peak_mb: The process's peak RSS after it (`peak_rss_mb`);
        a high-water mark, so it includes every step before.
    :param tracemalloc_peak_mb: The most memory Python held, as traced,
        at any point during the step; None when tracemalloc was not on.
    :param tracemalloc_held_mb: The traced memory still held once the step
        returned; None when tracemalloc was not on.
    """

    step: str
    seconds: float = 0.0
    rss_peak_mb: float = MEMORY_UNAVAILABLE
    tracemalloc_peak_mb: Optional[float] = None
    tracemalloc_held_mb: Optional[float] = None


@contextmanager
def track_step(step: str) -> Iterator[StepMemory]:
    """
    Measure one step: the block run inside the `with`.

    The tracemalloc peak is reset as the step starts, so it is this step's
    own; tracemalloc counts what Python and numpy allocate, not buffers
    pyarrow allocates itself, which only the RSS shows. Start tracemalloc
    beforehand to get its figures; without it they stay None. The record is
    filled when the block exits, also when it raises.

    :param step: The step's name.
    :return: The `StepMemory`, filled on exit.
    """
    record = StepMemory(step=step)
    tracing = tracemalloc.is_tracing()
    if tracing:
        tracemalloc.reset_peak()
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.seconds = time.perf_counter() - start
        record.rss_peak_mb = peak_rss_mb()
        if tracing:
            held, peak = tracemalloc.get_traced_memory()
            record.tracemalloc_peak_mb = peak / BYTES_TO_MB
            record.tracemalloc_held_mb = held / BYTES_TO_MB
