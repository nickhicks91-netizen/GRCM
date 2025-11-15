"""
Benchmarking script for EchoZero-Hybrid
"""
import torch
import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EchoZeroHybrid


def benchmark_forward_pass(model, batch_size, seq_len, num_iterations=100, warmup=10):
    """Benchmark forward pass speed"""
    device = next(model.parameters()).device

    # Create random input
    x = torch.randn(batch_size, seq_len).to(device)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []

    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            _ = model(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.time()
            times.append(end - start)

    times = np.array(times)

    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times),
        'throughput': batch_size / times.mean(),  # samples/sec
    }


def benchmark_memory(model, batch_size, seq_len):
    """Benchmark memory usage"""
    device = next(model.parameters()).device

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        x = torch.randn(batch_size, seq_len).to(device)

        # Forward pass
        with torch.no_grad():
            _ = model(x)

        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        return {
            'allocated_mb': memory_allocated,
            'reserved_mb': memory_reserved,
            'peak_mb': max_memory,
        }
    else:
        return {'note': 'Memory profiling only available on CUDA'}


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }


def main():
    """Run comprehensive benchmarks"""
    print("=" * 70)
    print("ECHOZERO-HYBRID BENCHMARKING")
    print("=" * 70)

    # Configuration
    configs = [
        {'dim': 128, 'seq_len': 512, 'batch_size': 16, 'name': 'Small'},
        {'dim': 256, 'seq_len': 1024, 'batch_size': 32, 'name': 'Medium'},
        {'dim': 512, 'seq_len': 2048, 'batch_size': 16, 'name': 'Large'},
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    for config in configs:
        print(f"\n{'-' * 70}")
        print(f"Configuration: {config['name']}")
        print(f"  Dim: {config['dim']}, Seq Len: {config['seq_len']}, Batch Size: {config['batch_size']}")
        print(f"{'-' * 70}")

        # Create model
        model = EchoZeroHybrid(
            dim=config['dim'],
            num_classes=2,
            reservoir_dim=config['dim'] * 2,
            reservoir_size=config['dim'],
        )
        model = model.to(device)
        model.eval()

        # Parameter count
        param_stats = count_parameters(model)
        print(f"\nParameters:")
        print(f"  Total:        {param_stats['total']:,}")
        print(f"  Trainable:    {param_stats['trainable']:,}")
        print(f"  Non-trainable: {param_stats['non_trainable']:,}")

        # Speed benchmark
        speed_stats = benchmark_forward_pass(
            model,
            config['batch_size'],
            config['seq_len'],
            num_iterations=50,
            warmup=5,
        )
        print(f"\nSpeed (forward pass):")
        print(f"  Mean:       {speed_stats['mean']*1000:.2f} ms")
        print(f"  Std:        {speed_stats['std']*1000:.2f} ms")
        print(f"  Min:        {speed_stats['min']*1000:.2f} ms")
        print(f"  Max:        {speed_stats['max']*1000:.2f} ms")
        print(f"  Throughput: {speed_stats['throughput']:.1f} samples/sec")

        # Memory benchmark
        mem_stats = benchmark_memory(model, config['batch_size'], config['seq_len'])
        print(f"\nMemory:")
        if 'allocated_mb' in mem_stats:
            print(f"  Allocated: {mem_stats['allocated_mb']:.2f} MB")
            print(f"  Reserved:  {mem_stats['reserved_mb']:.2f} MB")
            print(f"  Peak:      {mem_stats['peak_mb']:.2f} MB")
        else:
            print(f"  {mem_stats['note']}")

    print(f"\n{'=' * 70}")
    print("Benchmarking complete!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
