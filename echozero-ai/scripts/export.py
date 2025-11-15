"""
Export script for EchoZero-Hybrid (TorchScript & ONNX)
"""
import torch
import torch.onnx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EchoZeroHybrid


def export_torchscript(model, example_input, save_path):
    """Export model to TorchScript"""
    print(f"Exporting to TorchScript: {save_path}")

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Save
    traced_model.save(str(save_path))

    # Verify
    loaded = torch.jit.load(str(save_path))
    with torch.no_grad():
        original_out = model(example_input)
        traced_out = loaded(example_input)

    print("  TorchScript export successful!")
    print(f"  Output logits match: {torch.allclose(original_out['logits'], traced_out['logits'], atol=1e-5)}")


def export_onnx(model, example_input, save_path):
    """Export model to ONNX"""
    print(f"Exporting to ONNX: {save_path}")

    # Custom forward for ONNX (return only logits)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            return outputs['logits']

    wrapped_model = ONNXWrapper(model)

    # Export
    torch.onnx.export(
        wrapped_model,
        example_input,
        str(save_path),
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print("  ONNX export successful!")


def main():
    """Main export function"""
    print("=" * 70)
    print("ECHOZERO-HYBRID MODEL EXPORT")
    print("=" * 70)

    # Create output directory
    export_dir = Path('experiments')
    export_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("\nCreating model...")
    model = EchoZeroHybrid(
        dim=256,
        num_classes=2,
        reservoir_dim=512,
        reservoir_size=256,
    )
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create example input
    batch_size = 4
    seq_len = 1024
    example_input = torch.randn(batch_size, seq_len)

    print(f"\nExample input shape: {example_input.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(example_input)
        print(f"  Logits shape: {output['logits'].shape}")
        print(f"  Features shape: {output['features'].shape}")
        print(f"  Retained mass: {output['retained_mass']:.4f}")

    # Export TorchScript
    print("\n" + "-" * 70)
    ts_path = export_dir / 'echozero_ts.pt'
    export_torchscript(model, example_input, ts_path)

    # Export ONNX
    print("\n" + "-" * 70)
    onnx_path = export_dir / 'echozero.onnx'
    export_onnx(model, example_input, onnx_path)

    print("\n" + "=" * 70)
    print("Export complete!")
    print(f"  TorchScript: {ts_path}")
    print(f"  ONNX:        {onnx_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
