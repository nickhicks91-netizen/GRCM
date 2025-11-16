"""
Mega-Scale GRCM Test - Push to Memory Limits
Test with 1M-10M parameter models to prove scalability
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ModularGRCM, GRCMConfig, ModelComparator
from baselines.transformer_baseline import MultimodalTransformerBaseline, BaselineConfig


def main():
    print("=" * 70)
    print("MEGA-SCALE TEST - Pushing to Limits")
    print("=" * 70)

    # XXL configuration - ~2M parameters
    grcm_config_xxl = GRCMConfig(
        input_dim=12,
        freq_dim=128,
        memory_size=512,
        device='cpu'
    )

    baseline_config_xxl = BaselineConfig(
        input_dim=12,
        hidden_dim=2048,
        num_heads=16,
        num_layers=12,
        ff_dim=8192,
        device='cpu'
    )

    print("\n[1/2] Creating XXL models...")
    grcm_model = ModularGRCM(grcm_config_xxl)
    baseline_model = MultimodalTransformerBaseline(baseline_config_xxl)

    grcm_params = sum(p.numel() for p in grcm_model.parameters() if p.requires_grad)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)

    print(f"  GRCM:     {grcm_params:,} parameters")
    print(f"  Baseline: {baseline_params:,} parameters")
    print(f"  Ratio:    {baseline_params/grcm_params:.1f}x")

    print("\n[2/2] Running comparison (10 iterations)...")
    comparator = ModelComparator(grcm_model, baseline_model, device='cpu')

    result = comparator.compare_models(
        batch_size=2,  # Smaller batch to fit in memory
        num_iterations=10,
        platform='CPU'
    )

    print(f"\n{'='*70}")
    print("XXL RESULTS")
    print(f"{'='*70}")
    print(f"  Speedup:        {result.speedup:.2f}x")
    print(f"  FLOP reduction: {result.flop_reduction:.1f}%")
    print(f"  Energy savings: {result.energy_savings:.1f}%")
    print(f"  Memory: {result.grcm_memory_mb:.2f} MB (GRCM) vs {result.baseline_memory_mb:.2f} MB (Baseline)")

    print(f"\n🔥 At {grcm_params:,} parameters:")
    print(f"   {result.speedup:.1f}x faster than {baseline_params:,} parameter transformer")
    print(f"   {result.flop_reduction:.1f}% computational efficiency improvement")


if __name__ == "__main__":
    main()
