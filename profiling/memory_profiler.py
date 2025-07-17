#!/usr/bin/env python3
"""Memory profiling script for KNN implementations."""

import argparse
import os
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict

import psutil

try:
    from memory_profiler import profile  # type: ignore[attr-defined]

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import memray

    MEMRAY_AVAILABLE = True
except ImportError:
    MEMRAY_AVAILABLE = False


class MemoryProfiler:
    """Memory profiling utilities for KNN algorithms."""

    def __init__(self) -> None:
        """Initialize the memory profiler."""
        self.process = psutil.Process(os.getpid())
        self.tracemalloc_started = False

    def start_tracemalloc(self) -> None:
        """Start tracemalloc for detailed memory tracking."""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": self.process.memory_percent(),
        }

    def get_tracemalloc_stats(self) -> Dict[str, Any]:
        """Get tracemalloc statistics."""
        if not self.tracemalloc_started:
            return {}

        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024,
        }

    def profile_function(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Profile a function's memory usage."""
        self.start_tracemalloc()

        # Get initial memory
        initial_memory = self.get_memory_usage()

        # Run function
        result = func(*args, **kwargs)

        # Get final memory
        final_memory = self.get_memory_usage()
        tracemalloc_stats = self.get_tracemalloc_stats()

        return {
            "result": result,
            "memory_stats": {
                "initial": initial_memory,
                "final": final_memory,
                "delta_rss_mb": final_memory["rss_mb"] - initial_memory["rss_mb"],
                "tracemalloc": tracemalloc_stats,
            },
        }


def create_memory_profile_decorator() -> Callable[..., Any]:
    """Create a memory profiling decorator."""
    if MEMORY_PROFILER_AVAILABLE:
        return profile  # type: ignore[no-any-return]
    else:

        def dummy_profile(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return dummy_profile


# Memory profiling decorator
memory_profile = create_memory_profile_decorator()


def run_memray_profiling(
    func: Callable[..., Any], output_file: str, *args: Any, **kwargs: Any
) -> Any:
    """Run memray profiling on a function."""
    if not MEMRAY_AVAILABLE:
        print("Memray not available. Install with: pip install memray")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with memray.Tracker(output_path):
        result = func(*args, **kwargs)

    print(f"Memray profile saved to: {output_path}")
    print(f"View with: memray flamegraph {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory profiling utilities")
    parser.add_argument("--test", action="store_true", help="Run test profiling")
    args = parser.parse_args()

    if args.test:
        profiler = MemoryProfiler()


def test_memory_profiling() -> float:
    """Test the memory profiling functionality."""
    import numpy as np

    # Simulate some memory usage
    data = np.random.random((1000, 100))
    return float(data.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory profiling utilities")
    parser.add_argument("--test", action="store_true", help="Run test profiling")
    args = parser.parse_args()

    if args.test:
        profiler = MemoryProfiler()

        stats = profiler.profile_function(test_memory_profiling)
        print("Memory profiling test results:")
        print(f"Initial RSS: {stats['memory_stats']['initial']['rss_mb']:.2f} MB")
        print(f"Final RSS: {stats['memory_stats']['final']['rss_mb']:.2f} MB")
        print(f"Delta RSS: {stats['memory_stats']['delta_rss_mb']:.2f} MB")

        if stats["memory_stats"]["tracemalloc"]:
            tm_stats = stats["memory_stats"]["tracemalloc"]
            print(f"Tracemalloc Peak: {tm_stats['peak_mb']:.2f} MB")
    else:
        print("Memory profiling utilities loaded. Use --test to run a test.")
