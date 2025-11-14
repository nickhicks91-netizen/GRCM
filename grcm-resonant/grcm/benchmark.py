"""Benchmarking utilities for GRCM performance profiling."""
import torch
import time
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    num_iterations: int
    batch_size: int
    device: str
    optimization: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Get human-readable summary."""
        return f"""
{self.name}
{'=' * 60}
Mean Latency:     {self.mean_latency_ms:.2f} ± {self.std_latency_ms:.2f} ms
Median Latency:   {self.median_latency_ms:.2f} ms
Min/Max:          {self.min_latency_ms:.2f} / {self.max_latency_ms:.2f} ms
P95/P99:          {self.p95_latency_ms:.2f} / {self.p99_latency_ms:.2f} ms
Throughput:       {self.throughput_samples_per_sec:.1f} samples/sec
Batch Size:       {self.batch_size}
Iterations:       {self.num_iterations}
Device:           {self.device}
Optimization:     {self.optimization}
"""


class GRCMBenchmark:
    """
    Comprehensive benchmarking suite for GRCM.

    Measures:
        - Latency (mean, std, percentiles)
        - Throughput (samples/sec)
        - Memory usage
        - Coherence quality metrics
        - Phi stability

    Args:
        model: GRCM model or OptimizedGRCM instance
        device: torch device
        warmup_iterations: Number of warmup runs (default 10)
    """

    def __init__(
        self,
        model,
        device: Optional[torch.device] = None,
        warmup_iterations: int = 10
    ):
        self.model = model
        self.device = device or (next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu'))
        self.warmup_iterations = warmup_iterations
        self.results: List[BenchmarkResult] = []

    def _generate_inputs(self, batch_size: int):
        """Generate random inputs for testing."""
        return {
            'image_emb': torch.randn(batch_size, 512, device=self.device),
            'audio_emb': torch.randn(batch_size, 768, device=self.device),
            'action': torch.randn(batch_size, 4, device=self.device)
        }

    def _warmup(self):
        """Warmup runs to stabilize timing."""
        inputs = self._generate_inputs(1)
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model(**inputs)

    def benchmark_latency(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        name: str = "GRCM Forward Pass",
        optimization: str = "none"
    ) -> BenchmarkResult:
        """
        Benchmark forward pass latency.

        Args:
            batch_size: Batch size for testing
            num_iterations: Number of timing iterations
            name: Benchmark name
            optimization: Optimization method description

        Returns:
            BenchmarkResult with timing statistics
        """
        print(f"\nBenchmarking: {name}")
        print(f"  Batch size: {batch_size}, Iterations: {num_iterations}")

        # Warmup
        print("  Warming up...")
        self._warmup()

        # Benchmark
        print("  Running benchmark...")
        latencies = []
        inputs = self._generate_inputs(batch_size)

        for i in range(num_iterations):
            # Synchronize if CUDA
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                _ = self.model(**inputs)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

            if (i + 1) % 25 == 0:
                print(f"    Progress: {i + 1}/{num_iterations}")

        # Calculate statistics
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
        min_lat = min(latencies)
        max_lat = max(latencies)
        median_lat = statistics.median(latencies)

        sorted_lat = sorted(latencies)
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99_lat = sorted_lat[int(len(sorted_lat) * 0.99)]

        throughput = (batch_size * 1000) / mean_lat  # samples per second

        result = BenchmarkResult(
            name=name,
            mean_latency_ms=mean_lat,
            std_latency_ms=std_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            median_latency_ms=median_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            throughput_samples_per_sec=throughput,
            num_iterations=num_iterations,
            batch_size=batch_size,
            device=str(self.device),
            optimization=optimization
        )

        self.results.append(result)
        print(result.summary())

        return result

    def benchmark_batch_sizes(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        num_iterations: int = 50,
        optimization: str = "none"
    ) -> List[BenchmarkResult]:
        """
        Benchmark across different batch sizes.

        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Iterations per batch size
            optimization: Optimization description

        Returns:
            List of BenchmarkResults
        """
        results = []
        for bs in batch_sizes:
            result = self.benchmark_latency(
                batch_size=bs,
                num_iterations=num_iterations,
                name=f"GRCM Batch Size {bs}",
                optimization=optimization
            )
            results.append(result)

        return results

    def benchmark_coherence_quality(
        self,
        num_samples: int = 100,
        target_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Measure coherence quality metrics.

        Args:
            num_samples: Number of samples to test
            target_threshold: Coherence threshold (default 0.7)

        Returns:
            Dict with quality metrics
        """
        print(f"\nBenchmarking Coherence Quality ({num_samples} samples)")

        coherence_scores = []
        inputs = self._generate_inputs(1)

        with torch.no_grad():
            for _ in range(num_samples):
                result = self.model(**inputs)
                coh = result['coherence'].item() if hasattr(result['coherence'], 'item') else result['coherence']
                coherence_scores.append(coh)

        above_threshold = sum(1 for c in coherence_scores if c > target_threshold)
        percentage_above = (above_threshold / num_samples) * 100

        metrics = {
            'mean_coherence': statistics.mean(coherence_scores),
            'std_coherence': statistics.stdev(coherence_scores),
            'min_coherence': min(coherence_scores),
            'max_coherence': max(coherence_scores),
            'percentage_above_threshold': percentage_above,
            'target_threshold': target_threshold
        }

        print(f"  Mean Coherence:    {metrics['mean_coherence']:.3f} ± {metrics['std_coherence']:.3f}")
        print(f"  Range:             [{metrics['min_coherence']:.3f}, {metrics['max_coherence']:.3f}]")
        print(f"  Above {target_threshold}:      {percentage_above:.1f}%")

        return metrics

    def benchmark_phi_stability(
        self,
        num_iterations: int = 100,
        target_std: float = 0.2
    ) -> Dict[str, float]:
        """
        Measure phi (Φ) stability over time.

        Args:
            num_iterations: Number of iterations
            target_std: Target standard deviation (default 0.2)

        Returns:
            Dict with phi metrics
        """
        print(f"\nBenchmarking Phi Stability ({num_iterations} iterations)")

        phi_values = []
        inputs = self._generate_inputs(1)

        with torch.no_grad():
            for _ in range(num_iterations):
                result = self.model(**inputs)
                phi = result['phi']
                phi_values.append(phi)

        metrics = {
            'mean_phi': statistics.mean(phi_values),
            'std_phi': statistics.stdev(phi_values),
            'min_phi': min(phi_values),
            'max_phi': max(phi_values),
            'stable': statistics.stdev(phi_values) < target_std,
            'target_std': target_std
        }

        status = "✓ STABLE" if metrics['stable'] else "✗ UNSTABLE"
        print(f"  Mean Phi:          {metrics['mean_phi']:.3f} ± {metrics['std_phi']:.3f} {status}")
        print(f"  Range:             [{metrics['min_phi']:.3f}, {metrics['max_phi']:.3f}]")
        print(f"  Target Std:        < {target_std}")

        return metrics

    def memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage.

        Returns:
            Dict with memory statistics (MB)
        """
        print("\nMemory Usage")

        # Model parameters
        if hasattr(self.model, 'parameters'):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            param_memory_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        else:
            # OptimizedGRCM wrapper
            total_params = sum(p.numel() for p in self.model.original_model.parameters())
            trainable_params = sum(p.numel() for p in self.model.original_model.parameters() if p.requires_grad)
            param_memory_mb = sum(p.numel() * p.element_size() for p in self.model.original_model.parameters()) / (1024 ** 2)

        # Activation memory (estimated for batch=1)
        activation_memory_mb = 5.0  # Rough estimate

        metrics = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_memory_mb': param_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'total_memory_mb': param_memory_mb + activation_memory_mb
        }

        print(f"  Total Parameters:  {total_params:,}")
        print(f"  Trainable:         {trainable_params:,}")
        print(f"  Param Memory:      {param_memory_mb:.2f} MB")
        print(f"  Activation Memory: {activation_memory_mb:.2f} MB (batch=1)")
        print(f"  Total Memory:      {metrics['total_memory_mb']:.2f} MB")

        return metrics

    def full_benchmark(
        self,
        optimization: str = "none",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            optimization: Optimization description
            save_path: Optional path to save results JSON

        Returns:
            Dict with all benchmark results
        """
        print("=" * 60)
        print("GRCM FULL BENCHMARK SUITE")
        print("=" * 60)

        results = {
            'optimization': optimization,
            'device': str(self.device),
            'latency': {},
            'batch_sizes': [],
            'coherence_quality': {},
            'phi_stability': {},
            'memory': {}
        }

        # 1. Single-sample latency
        lat_result = self.benchmark_latency(
            batch_size=1,
            num_iterations=100,
            name="Single Sample Latency",
            optimization=optimization
        )
        results['latency'] = lat_result.to_dict()

        # 2. Batch sizes
        batch_results = self.benchmark_batch_sizes(
            batch_sizes=[1, 2, 4, 8],
            num_iterations=50,
            optimization=optimization
        )
        results['batch_sizes'] = [r.to_dict() for r in batch_results]

        # 3. Coherence quality
        results['coherence_quality'] = self.benchmark_coherence_quality(num_samples=100)

        # 4. Phi stability
        results['phi_stability'] = self.benchmark_phi_stability(num_iterations=100)

        # 5. Memory
        results['memory'] = self.memory_usage()

        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {save_path}")

        return results

    def compare_optimizations(
        self,
        models: Dict[str, Any],
        num_iterations: int = 50
    ) -> None:
        """
        Compare multiple optimization strategies.

        Args:
            models: Dict mapping optimization name to model instance
            num_iterations: Iterations per model
        """
        print("=" * 60)
        print("OPTIMIZATION COMPARISON")
        print("=" * 60)

        comparison = []

        for name, model in models.items():
            benchmark = GRCMBenchmark(model, self.device, warmup_iterations=5)
            result = benchmark.benchmark_latency(
                batch_size=1,
                num_iterations=num_iterations,
                name=name,
                optimization=name
            )
            comparison.append(result)

        # Print comparison table
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Optimization':<25} {'Mean (ms)':<12} {'Throughput (samp/s)':<20}")
        print("-" * 60)

        for result in comparison:
            print(f"{result.name:<25} {result.mean_latency_ms:<12.2f} {result.throughput_samples_per_sec:<20.1f}")

        # Calculate speedups
        baseline = comparison[0]
        print("\n" + "=" * 60)
        print("SPEEDUP vs BASELINE")
        print("=" * 60)

        for result in comparison[1:]:
            speedup = baseline.mean_latency_ms / result.mean_latency_ms
            print(f"{result.name:<25} {speedup:.2f}x faster")


if __name__ == "__main__":
    print("GRCM Benchmarking Module")
    print("=" * 50)
    print("\nAvailable classes:")
    print("  - BenchmarkResult: Data class for results")
    print("  - GRCMBenchmark: Comprehensive benchmarking suite")
