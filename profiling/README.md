# Profiling Results

This directory contains profiling results and utilities for the baseline KNN implementation.

## Profiling Tools

### Memory Profiling (`memory_profiler.py`)
- **Purpose**: Profile memory usage of KNN implementations
- **Features**: 
  - Process memory tracking (RSS, VMS)
  - Tracemalloc integration for detailed memory tracking
  - Memray support for advanced memory profiling
- **Usage**: `python profiling/memory_profiler.py --test`

### Speed Profiling (`speed_profiler.py`)
- **Purpose**: Profile execution speed and performance
- **Features**:
  - Function timing
  - cProfile integration
  - Line-by-line profiling (line_profiler)
  - Benchmarking utilities
- **Usage**: `python profiling/speed_profiler.py --test --benchmark`

### Comprehensive Profiling (`profile_runner.py`)
- **Purpose**: Run comprehensive profiling combining memory and speed analysis
- **Features**:
  - Full profiling pipeline
  - Comparison benchmarks
  - Automated report generation
- **Usage**: `python profiling/profile_runner.py --test`

## Available Profiling Dependencies

The following profiling tools are available via the `profiling` optional dependency group:

- `memory-profiler`: Line-by-line memory usage
- `psutil`: System and process utilities
- `line-profiler`: Line-by-line timing
- `py-spy`: Sampling profiler
- `memray`: Advanced memory profiler
- `scalene`: CPU, memory, and GPU profiler

Install with: `pip install -e ".[profiling]"`

## Benchmark Dependencies

Additional benchmarking tools available via the `benchmark` optional dependency:

- `pytest-benchmark`: Pytest benchmarking plugin
- `pympler`: Advanced memory analysis
- `tracemalloc-tools`: Tracemalloc utilities

Install with: `pip install -e ".[benchmark]"`

## Usage Examples

### Basic Memory Profiling
```python
from profiling.memory_profiler import MemoryProfiler, memory_profile

profiler = MemoryProfiler()

@memory_profile
def my_knn_function():
    # Your KNN implementation
    pass

# Or profile directly
stats = profiler.profile_function(my_knn_function)
```

### Basic Speed Profiling
```python
from profiling.speed_profiler import SpeedProfiler, timing_decorator

profiler = SpeedProfiler()

@timing_decorator
def my_knn_function():
    # Your KNN implementation
    pass

# Or benchmark
stats = profiler.benchmark_function(my_knn_function, iterations=10)
```

### Comprehensive Profiling
```python
from profiling.profile_runner import KNNProfiler

profiler = KNNProfiler()
results = profiler.full_profile(my_knn_function, "my_knn")
```

## Output Files

Profiling results are saved in the `profiling/results/` directory:

- `*.prof`: cProfile binary output (use with `python -m pstats`)
- `*_stats.txt`: Human-readable cProfile statistics
- `*_line_profile.txt`: Line profiler output
- `*_summary.txt`: Profiling summary
- `benchmark_comparison.txt`: Benchmark comparison results
- `*.bin`: Memray binary output (use with `memray flamegraph`)

## Profiling Best Practices

1. **Warm-up runs**: Run functions once before profiling to account for JIT compilation
2. **Multiple iterations**: Use benchmarking for statistical significance
3. **Consistent environment**: Profile in similar conditions to production
4. **Memory baseline**: Consider baseline memory usage of your environment
5. **Data size scaling**: Test with various data sizes to understand complexity
