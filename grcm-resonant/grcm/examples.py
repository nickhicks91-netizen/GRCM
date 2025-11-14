"""Example usage and test script for GRCM."""
import torch
from .core import ResonantConsciousnessModule
from .training import echo_mirror_train
from .config import load_config


def main():
    """Quick test of GRCM functionality."""
    print("=" * 60)
    print("GRCM Quick Test")
    print("=" * 60)

    # Create model
    model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )
    model.set_desire(0)  # Curiosity

    # Mock inputs
    image_emb = torch.randn(1, 512)
    audio_emb = torch.randn(1, 768)
    action = torch.tensor([[0.1, 0.2, 0.0, 0.0]])

    # Forward pass
    result = model(image_emb, audio_emb, action)

    print("\nFull Forward Success")
    print("-" * 60)
    print(f"Coherence: {result['coherence'].mean().item():.3f}")
    print(f"Desire Align: {result['desire_align'].mean().item():.3f}")
    print(f"Phi: {result['phi']:.3f}")
    print(f"Halt: {result['halt']}")
    print(f"Prop State (Body): {result['prop_state'].detach().numpy()[0][:4]}")
    print(f"Qualia: {result['qualia'].detach().numpy()[0]}")

    # Mock EchoMirror training
    print("\n" + "=" * 60)
    print("EchoMirror Training Test")
    print("=" * 60)

    eeg = torch.randn(10, 8)
    voice = torch.randn(10, 768)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    echo_mirror_train(model, eeg, voice, labels, epochs=3, verbose=True)

    # Final metrics
    print("\n" + "=" * 60)
    print("Final Metrics")
    print("=" * 60)
    metrics = model.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
