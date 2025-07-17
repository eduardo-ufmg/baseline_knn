#!/usr/bin/env python3
"""Speed profiling script for KNN implementations."""

import argparse
import cProfile
import functools
import pstats
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from line_profiler import LineProfiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False


class SpeedProfiler:
    """Speed profiling utilities for KNN algorithms."""

    def __init__(self, output_dir: str = "profiling/results") -> None:
        """Initialize the speed profiler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def time_function(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        return {
            "result": result,
            "execution_time": end_time - start_time,
            "execution_time_ms": (end_time - start_time) * 1000,
        }

    def profile_with_cprofile(
        self, func: Callable[..., Any], filename: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Profile function with cProfile."""
        output_file = self.output_dir / f"{filename}.prof"

        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()
        profiler.dump_stats(str(output_file))

        # Create human-readable stats
        stats_file = self.output_dir / f"{filename}_stats.txt"
        with open(stats_file, "w") as f:
            stats = pstats.Stats(str(output_file))
            stats.sort_stats("cumulative")
            # Redirect stdout to file for print_stats
            import sys

            old_stdout = sys.stdout
            sys.stdout = f
            stats.print_stats()
            sys.stdout = old_stdout

        print(f"cProfile results saved to: {output_file}")
        print(f"Human-readable stats saved to: {stats_file}")
        print(f"View with: python -m pstats {output_file}")

        return result

    def profile_with_line_profiler(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Profile function with line_profiler."""
        if not LINE_PROFILER_AVAILABLE:
            print(
                "line_profiler not available. Install with: pip install line_profiler"
            )
            return func(*args, **kwargs)

        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.enable_by_count()

        result = func(*args, **kwargs)

        profiler.disable_by_count()

        # Save results
        output_file = self.output_dir / f"{func.__name__}_line_profile.txt"
        with open(output_file, "w") as f:
            profiler.print_stats(stream=f)

        print(f"Line profiler results saved to: {output_file}")
        return result

    def benchmark_function(
        self, func: Callable[..., Any], iterations: int = 10, *args: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Benchmark a function over multiple iterations."""
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "iterations": iterations,
        }


def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Time function execution."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result

    return wrapper


def profile_decorator(output_dir: str = "profiling/results") -> Callable[..., Any]:
    """Profile function with cProfile."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = SpeedProfiler(output_dir)
            return profiler.profile_with_cprofile(func, func.__name__, *args, **kwargs)

        return wrapper

    return decorator


def line_profile_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Profile function with line_profiler."""
    if not LINE_PROFILER_AVAILABLE:
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        profiler = SpeedProfiler()
        return profiler.profile_with_line_profiler(func, *args, **kwargs)

    return wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speed profiling utilities")
    parser.add_argument("--test", action="store_true", help="Run test profiling")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark test")
    args = parser.parse_args()

    if args.test or args.benchmark:
        profiler = SpeedProfiler()

        @timing_decorator
        def test_function() -> Any:
            """Test speed profiling functionality."""
            import numpy as np

            # Simulate some computation
            data = np.random.random((1000, 100))
            return np.linalg.norm(data, axis=1)

        if args.test:
            print("Running speed profiling test...")
            stats = profiler.time_function(test_function)
            print(f"Execution time: {stats['execution_time']:.4f} seconds")

            # Profile with cProfile
            profiler.profile_with_cprofile(test_function, "test_function")

        if args.benchmark:
            print("Running benchmark test...")
            benchmark_stats = profiler.benchmark_function(test_function, iterations=5)
            print(f"Mean execution time: {benchmark_stats['mean_time']:.4f} seconds")
            print(f"Min execution time: {benchmark_stats['min_time']:.4f} seconds")
            print(f"Max execution time: {benchmark_stats['max_time']:.4f} seconds")
    else:
        print(
            "Speed profiling utilities loaded. Use --test or --benchmark to run tests."
        )
