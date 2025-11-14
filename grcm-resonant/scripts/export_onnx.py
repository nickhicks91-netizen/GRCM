#!/usr/bin/env python3
"""
ONNX Export Script for GRCM

Exports GRCM to ONNX format with dynamic batch axes and validation.

Usage:
    python scripts/export_onnx.py --output grcm_model.onnx
    python scripts/export_onnx.py --output grcm_model.onnx --simplify --test
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ResonantConsciousnessModule
from grcm.optimization import export_to_onnx, test_onnx_inference


def main():
    parser = argparse.ArgumentParser(description='Export GRCM to ONNX')
    parser.add_argument(
        '--output',
        type=str,
        default='grcm_model.onnx',
        help='Output ONNX file path'
    )
    parser.add_argument(
        '--input-dim',
        type=int,
        default=15,
        help='Input dimension (default 15)'
    )
    parser.add_argument(
        '--freq-dim',
        type=int,
        default=8,
        help='Frequency dimension (default 8)'
    )
    parser.add_argument(
        '--memory-size',
        type=int,
        default=32,
        help='Memory size (default 32)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=18,
        help='ONNX opset version (default 18)'
    )
    parser.add_argument(
        '--no-dynamic-batch',
        action='store_true',
        help='Disable dynamic batch dimension'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX graph (requires onnx-simplifier)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test ONNX model with ONNXRuntime'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=5,
        help='Number of test samples (default 5)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GRCM ONNX Export")
    print("=" * 60)
    print(f"\nModel Configuration:")
    print(f"  Input Dim:     {args.input_dim}")
    print(f"  Freq Dim:      {args.freq_dim}")
    print(f"  Memory Size:   {args.memory_size}")
    print(f"\nExport Settings:")
    print(f"  Output:        {args.output}")
    print(f"  Opset:         {args.opset}")
    print(f"  Dynamic Batch: {not args.no_dynamic_batch}")
    print(f"  Simplify:      {args.simplify}")

    # Create model
    print("\nCreating GRCM model...")
    model = ResonantConsciousnessModule(
        input_dim=args.input_dim,
        freq_dim=args.freq_dim,
        memory_size=args.memory_size
    )
    print("✓ Model created")

    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        export_to_onnx(
            model,
            output_path=args.output,
            opset_version=args.opset,
            dynamic_batch=not args.no_dynamic_batch,
            simplify=args.simplify
        )
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return 1

    # Test if requested
    if args.test:
        print("\nTesting ONNX model...")
        test_onnx_inference(args.output, num_samples=args.test_samples)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nONNX model saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Test inference: python -c \"from grcm.optimization import test_onnx_inference; test_onnx_inference('{args.output}')\"")
    print(f"  2. Convert to TensorRT: See docs/TENSORRT_GUIDE.md")
    print(f"  3. Deploy with ONNXRuntime for cross-platform inference")

    return 0


if __name__ == "__main__":
    sys.exit(main())
