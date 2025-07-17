#!/usr/bin/env python3
"""Comprehensive profiling runner for KNN implementations."""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from .memory_profiler import MemoryProfiler  # noqa: E402
from .speed_profiler import SpeedProfiler  # noqa: E402


class KNNProfiler:
    """Comprehensive profiler for KNN implementations."""

    def __init__(self, output_dir: str = "profiling/results") -> None:
        """Initialize the KNN profiler."""
        self.memory_profiler = MemoryProfiler()
        self.speed_profiler = SpeedProfiler(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def full_profile(
        self, func: Callable[..., Any], name: str, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Run comprehensive profiling on a function."""
        print(f"Running comprehensive profiling for {name}...")

        # Speed profiling
        print("  - Speed profiling...")
        speed_stats = self.speed_profiler.time_function(func, *args, **kwargs)

        # Memory profiling
        print("  - Memory profiling...")
        memory_stats = self.memory_profiler.profile_function(func, *args, **kwargs)

        # cProfile
        print("  - cProfile profiling...")
        self.speed_profiler.profile_with_cprofile(func, name, *args, **kwargs)

        # Line profiling (if available)
        print("  - Line profiling...")
        self.speed_profiler.profile_with_line_profiler(func, *args, **kwargs)

        results = {
            "function_name": name,
            "speed_stats": speed_stats,
            "memory_stats": memory_stats,
        }

        # Save results summary
        summary_file = self.output_dir / f"{name}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Profiling Summary for {name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Execution Time: {speed_stats['execution_time']:.4f} seconds\n")
            f.write(
                f"Memory Delta RSS: "
                f"{memory_stats['memory_stats']['delta_rss_mb']:.2f} MB\n"
            )
            if memory_stats["memory_stats"]["tracemalloc"]:
                tm = memory_stats["memory_stats"]["tracemalloc"]
                f.write(f"Peak Memory (tracemalloc): {tm['peak_mb']:.2f} MB\n")

        print(f"Summary saved to: {summary_file}")
        return results

    def benchmark_comparison(
        self,
        functions_dict: Dict[str, Callable[..., Any]],
        iterations: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Benchmark multiple functions for comparison."""
        print(f"Running benchmark comparison with {iterations} iterations...")

        results = {}
        for name, func in functions_dict.items():
            print(f"Benchmarking {name}...")
            benchmark_stats = self.speed_profiler.benchmark_function(
                func, iterations, *args, **kwargs
            )
            results[name] = benchmark_stats

        # Create comparison report
        report_file = self.output_dir / "benchmark_comparison.txt"
        with open(report_file, "w") as f:
            f.write("Benchmark Comparison Report\n")
            f.write("=" * 50 + "\n\n")

            for name, stats in results.items():
                f.write(f"{name}:\n")
                f.write(f"  Mean time: {stats['mean_time']:.4f} seconds\n")
                f.write(f"  Min time:  {stats['min_time']:.4f} seconds\n")
                f.write(f"  Max time:  {stats['max_time']:.4f} seconds\n")
                f.write(f"  Total time: {stats['total_time']:.4f} seconds\n\n")

            # Find fastest
            fastest = min(results.items(), key=lambda x: x[1]["mean_time"])
            f.write(f"Fastest: {fastest[0]} ({fastest[1]['mean_time']:.4f}s mean)\n")

        print(f"Comparison report saved to: {report_file}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive KNN profiling")
    parser.add_argument("--test", action="store_true", help="Run test profiling")
    parser.add_argument(
        "--output-dir", default="profiling/results", help="Output directory"
    )
    args = parser.parse_args()

    if args.test:
        profiler = KNNProfiler(args.output_dir)

        def test_knn_function() -> Any:
            """Dummy KNN function for testing profiling."""
            import numpy as np
            from sklearn.neighbors import NearestNeighbors

            # Generate test data
            X = np.random.random((1000, 10))
            query = np.random.random((50, 10))

            # Fit and query
            nn = NearestNeighbors(n_neighbors=5, algorithm="auto")
            nn.fit(X)
            distances, indices = nn.kneighbors(query)

            return distances, indices

        # Run comprehensive profiling
        results = profiler.full_profile(test_knn_function, "test_knn")

        print("\nProfiling completed!")
        print(f"Execution time: {results['speed_stats']['execution_time']:.4f} seconds")
        print(
            f"Memory delta: "
            f"{results['memory_stats']['memory_stats']['delta_rss_mb']:.2f} MB"
        )
    else:
        print("KNN profiling utilities loaded. Use --test to run a test.")
