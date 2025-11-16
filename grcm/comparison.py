"""
GRCM Comparison Framework
Side-by-side benchmarking of GRCM vs transformer baselines
"""
import torch
import torch.nn as nn
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json

from .core import ModularGRCM
from .config import GRCMConfig


@dataclass
class ComparisonResult:
    """Results from comparing two models"""
    grcm_latency_ms: float
    baseline_latency_ms: float
    speedup: float

    grcm_memory_mb: float
    baseline_memory_mb: float
    memory_reduction: float

    grcm_flops: int
    baseline_flops: int
    flop_reduction: float

    grcm_energy_mj: float
    baseline_energy_mj: float
    energy_savings: float

    grcm_params: int
    baseline_params: int

    batch_size: int
    num_iterations: int


class ModelComparator:
    """
    Compare GRCM with baseline transformer models

    Measures:
    - Latency (inference time)
    - Throughput (samples/sec)
    - Memory usage (parameters + activations)
    - FLOPs (floating point operations)
    - Energy consumption (estimated)
    """

    def __init__(
        self,
        grcm_model: ModularGRCM,
        baseline_model: nn.Module,
        device: str = "cpu"
    ):
        self.grcm_model = grcm_model.to(device)
        self.baseline_model = baseline_model.to(device)
        self.device = device

        self.grcm_model.eval()
        self.baseline_model.eval()

    def _prepare_inputs(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Prepare dummy inputs for both models"""
        image_emb = torch.randn(batch_size, 512, device=self.device)
        audio_emb = torch.randn(batch_size, 768, device=self.device)
        action = torch.randn(batch_size, 4, device=self.device)
        return image_emb, audio_emb, action

    def _measure_latency(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> List[float]:
        """
        Measure inference latency

        Args:
            model: Model to benchmark
            inputs: Input tensors
            num_iterations: Number of timing runs
            warmup: Number of warmup iterations

        Returns:
            List of latencies in milliseconds
        """
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*inputs)

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(*inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        return latencies

    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _measure_memory(self, model: nn.Module) -> float:
        """
        Measure model memory usage in MB

        Returns:
            Memory usage in megabytes
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size
        return total_size / (1024 ** 2)  # Convert to MB

    def _count_flops(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        use_profiler: bool = True
    ) -> int:
        """
        Count FLOPs for forward pass

        Args:
            model: Model to analyze
            inputs: Input tensors
            use_profiler: Use torch.profiler (requires PyTorch 2.0+)

        Returns:
            Total FLOPs
        """
        if use_profiler and hasattr(torch.profiler, 'profile'):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_flops=True
            ) as prof:
                with torch.no_grad():
                    _ = model(*inputs)

            # Get total FLOPs from profiler
            total_flops = sum(
                [event.flops for event in prof.key_averages() if event.flops > 0]
            )
            return int(total_flops)
        else:
            # Fallback: Estimate based on parameters and operations
            return self._estimate_flops_manual(model, inputs)

    def _estimate_flops_manual(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...]
    ) -> int:
        """
        Manual FLOP estimation (fallback)

        Rough estimate based on:
        - Linear layers: 2 * in_features * out_features * batch_size
        - MultiheadAttention: ~4 * d_model^2 * seq_len * batch_size
        """
        batch_size = inputs[0].size(0)
        total_flops = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                # Multiply-add counts as 2 FLOPs
                total_flops += 2 * in_features * out_features * batch_size

            elif isinstance(module, nn.MultiheadAttention):
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                # Rough estimate: Q*K^T + softmax + attention*V
                # Per head: ~4 * (d/h)^2 per position
                # Total: ~4 * d^2 * seq_len (simplified)
                seq_len = 3  # Approximate for multimodal (3 modalities)
                total_flops += 4 * (embed_dim ** 2) * seq_len * batch_size

            elif isinstance(module, nn.GRUCell):
                hidden_size = module.hidden_size
                input_size = module.input_size
                # GRU has 3 gates, each with 2 matrix multiplications
                total_flops += 6 * (input_size + hidden_size) * hidden_size * batch_size

        return int(total_flops)

    def _estimate_energy(
        self,
        flops: int,
        latency_ms: float,
        platform: str = "A100"
    ) -> float:
        """
        Estimate energy consumption in millijoules

        Args:
            flops: Total floating point operations
            latency_ms: Inference latency in milliseconds
            platform: Hardware platform (A100, V100, CPU)

        Returns:
            Energy in millijoules (mJ)
        """
        # Energy per FLOP varies by hardware (in picojoules/FLOP)
        energy_per_flop = {
            'A100': 1.0,    # ~300W / 312 TFLOPS = ~1 pJ/FLOP
            'V100': 2.0,    # ~300W / 125 TFLOPS = ~2.4 pJ/FLOP
            'CPU': 100.0,   # ~100W / 1 TFLOPS = ~100 pJ/FLOP
        }

        pj_per_flop = energy_per_flop.get(platform, 10.0)

        # Energy (mJ) = FLOPs * pJ/FLOP / 1e9
        energy_mj = (flops * pj_per_flop) / 1e9

        return energy_mj

    def compare_models(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        platform: str = "A100"
    ) -> ComparisonResult:
        """
        Full comparison of GRCM vs baseline

        Args:
            batch_size: Batch size for testing
            num_iterations: Number of timing iterations
            platform: Hardware platform for energy estimation

        Returns:
            ComparisonResult with all metrics
        """
        print(f"\n{'='*70}")
        print(f"Model Comparison: GRCM vs Transformer Baseline")
        print(f"{'='*70}")
        print(f"Batch size: {batch_size}")
        print(f"Iterations: {num_iterations}")
        print(f"Platform: {platform}")

        # Prepare inputs
        inputs = self._prepare_inputs(batch_size)

        # 1. Measure latency
        print(f"\n[1/5] Measuring latency...")
        grcm_latencies = self._measure_latency(
            self.grcm_model, inputs, num_iterations
        )
        baseline_latencies = self._measure_latency(
            self.baseline_model, inputs, num_iterations
        )

        grcm_latency = statistics.mean(grcm_latencies)
        baseline_latency = statistics.mean(baseline_latencies)
        speedup = baseline_latency / grcm_latency

        print(f"  GRCM:     {grcm_latency:.2f}ms")
        print(f"  Baseline: {baseline_latency:.2f}ms")
        print(f"  Speedup:  {speedup:.2f}x")

        # 2. Measure memory
        print(f"\n[2/5] Measuring memory usage...")
        grcm_memory = self._measure_memory(self.grcm_model)
        baseline_memory = self._measure_memory(self.baseline_model)
        memory_reduction = (1 - grcm_memory / baseline_memory) * 100

        print(f"  GRCM:      {grcm_memory:.2f} MB")
        print(f"  Baseline:  {baseline_memory:.2f} MB")
        print(f"  Reduction: {memory_reduction:.1f}%")

        # 3. Count parameters
        print(f"\n[3/5] Counting parameters...")
        grcm_params = self._count_parameters(self.grcm_model)
        baseline_params = self._count_parameters(self.baseline_model)

        print(f"  GRCM:     {grcm_params:,}")
        print(f"  Baseline: {baseline_params:,}")

        # 4. Count FLOPs
        print(f"\n[4/5] Counting FLOPs...")
        try:
            grcm_flops = self._count_flops(self.grcm_model, inputs)
            baseline_flops = self._count_flops(self.baseline_model, inputs)
        except Exception as e:
            print(f"  Warning: FLOP profiling failed ({e}), using manual estimation")
            grcm_flops = self._estimate_flops_manual(self.grcm_model, inputs)
            baseline_flops = self._estimate_flops_manual(self.baseline_model, inputs)

        flop_reduction = (1 - grcm_flops / baseline_flops) * 100

        print(f"  GRCM:      {grcm_flops:,} FLOPs")
        print(f"  Baseline:  {baseline_flops:,} FLOPs")
        print(f"  Reduction: {flop_reduction:.1f}%")

        # 5. Estimate energy
        print(f"\n[5/5] Estimating energy consumption...")
        grcm_energy = self._estimate_energy(grcm_flops, grcm_latency, platform)
        baseline_energy = self._estimate_energy(
            baseline_flops, baseline_latency, platform
        )
        energy_savings = (1 - grcm_energy / baseline_energy) * 100

        print(f"  GRCM:     {grcm_energy:.3f} mJ")
        print(f"  Baseline: {baseline_energy:.3f} mJ")
        print(f"  Savings:  {energy_savings:.1f}%")

        # Create result
        result = ComparisonResult(
            grcm_latency_ms=grcm_latency,
            baseline_latency_ms=baseline_latency,
            speedup=speedup,
            grcm_memory_mb=grcm_memory,
            baseline_memory_mb=baseline_memory,
            memory_reduction=memory_reduction,
            grcm_flops=grcm_flops,
            baseline_flops=baseline_flops,
            flop_reduction=flop_reduction,
            grcm_energy_mj=grcm_energy,
            baseline_energy_mj=baseline_energy,
            energy_savings=energy_savings,
            grcm_params=grcm_params,
            baseline_params=baseline_params,
            batch_size=batch_size,
            num_iterations=num_iterations
        )

        return result

    def compare_batch_scaling(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        num_iterations: int = 50
    ) -> Dict[int, ComparisonResult]:
        """
        Compare models across different batch sizes

        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Iterations per batch size

        Returns:
            Dictionary mapping batch_size -> ComparisonResult
        """
        print(f"\n{'='*70}")
        print(f"Batch Scaling Comparison")
        print(f"{'='*70}")

        results = {}

        for batch_size in batch_sizes:
            print(f"\n--- Batch Size: {batch_size} ---")
            result = self.compare_models(batch_size, num_iterations)
            results[batch_size] = result

        return results

    def print_summary(self, result: ComparisonResult):
        """Print comparison summary"""
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}")

        print(f"\nEfficiency Gains:")
        print(f"  ⚡ Latency:  {result.speedup:.2f}x faster")
        print(f"  💾 Memory:   {result.memory_reduction:.1f}% reduction")
        print(f"  🔢 FLOPs:    {result.flop_reduction:.1f}% reduction")
        print(f"  ⚡ Energy:   {result.energy_savings:.1f}% savings")

        print(f"\nModel Sizes:")
        print(f"  GRCM:     {result.grcm_params:,} parameters")
        print(f"  Baseline: {result.baseline_params:,} parameters")

        print(f"\n{'='*70}")

        # Datacenter extrapolation
        if result.speedup > 1.0:
            self._print_datacenter_impact(result)

    def _print_datacenter_impact(self, result: ComparisonResult):
        """Calculate and print datacenter-scale impact"""
        print(f"\nDatacenter Impact (1 GW facility):")
        print(f"{'='*70}")

        # Assumptions
        datacenter_power_mw = 1000  # 1 GW = 1000 MW
        hours_per_year = 8760
        cost_per_kwh = 0.10  # $0.10/kWh

        # Energy savings
        baseline_energy_kwh = (
            result.baseline_energy_mj / 1000 / 3600  # mJ -> kWh per inference
        )
        grcm_energy_kwh = result.grcm_energy_mj / 1000 / 3600

        # Assume 1 billion inferences per day
        inferences_per_day = 1e9
        inferences_per_year = inferences_per_day * 365

        baseline_annual_kwh = baseline_energy_kwh * inferences_per_year
        grcm_annual_kwh = grcm_energy_kwh * inferences_per_year

        savings_kwh = baseline_annual_kwh - grcm_annual_kwh
        savings_mwh = savings_kwh / 1000
        savings_dollars = savings_kwh * cost_per_kwh

        print(f"  Assumptions:")
        print(f"    - 1 billion inferences/day")
        print(f"    - ${cost_per_kwh:.2f}/kWh electricity cost")
        print(f"")
        print(f"  Annual Energy:")
        print(f"    - Baseline: {baseline_annual_kwh/1e6:.1f} million kWh")
        print(f"    - GRCM:     {grcm_annual_kwh/1e6:.1f} million kWh")
        print(f"    - Savings:  {savings_mwh/1e3:.1f} GWh/year")
        print(f"")
        print(f"  Annual Cost:")
        print(f"    - Savings: ${savings_dollars/1e6:.1f}M/year")
        print(f"")
        print(f"  Infrastructure Impact:")
        print(f"    - Power reduction: {result.energy_savings:.1f}%")
        print(f"    - Equivalent to shutting down {result.energy_savings/100:.2f}x 1GW datacenter")

        print(f"\n{'='*70}")

    def export_results(
        self,
        result: ComparisonResult,
        output_path: str
    ):
        """Export comparison results to JSON"""
        result_dict = {
            'grcm': {
                'latency_ms': result.grcm_latency_ms,
                'memory_mb': result.grcm_memory_mb,
                'flops': result.grcm_flops,
                'energy_mj': result.grcm_energy_mj,
                'parameters': result.grcm_params
            },
            'baseline': {
                'latency_ms': result.baseline_latency_ms,
                'memory_mb': result.baseline_memory_mb,
                'flops': result.baseline_flops,
                'energy_mj': result.baseline_energy_mj,
                'parameters': result.baseline_params
            },
            'efficiency_gains': {
                'speedup': result.speedup,
                'memory_reduction_pct': result.memory_reduction,
                'flop_reduction_pct': result.flop_reduction,
                'energy_savings_pct': result.energy_savings
            },
            'test_config': {
                'batch_size': result.batch_size,
                'num_iterations': result.num_iterations
            }
        }

        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"\n[Export] Results saved to {output_path}")
