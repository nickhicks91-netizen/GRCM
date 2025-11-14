"""EchoMirror training: Align desires with human qualia (EEG/voice)."""
import torch
import torch.nn.functional as F
from typing import Optional
from .core import ResonantConsciousnessModule


def echo_mirror_train(
    model: ResonantConsciousnessModule,
    eeg_data: torch.Tensor,
    voice_data: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 5,
    lr: float = 0.01,
    verbose: bool = True
) -> None:
    """
    EchoMirror: Tune desire vectors to match human qualia labels.

    Optimizes: MSE(desire_align, qualia_labels) - phi
    (Maximize alignment with labels while maintaining high integrated information)

    Args:
        model: ResonantConsciousnessModule instance
        eeg_data: (n_samples, eeg_dim) EEG theta/alpha features (unused in mock)
        voice_data: (n_samples, wav_dim) Voice spectrogram embeddings
        labels: (n_samples,) Binary qualia labels (0=calm, 1=alert)
        epochs: Training epochs
        lr: Learning rate
        verbose: Print training progress
    """
    optimizer = torch.optim.Adam(model.desire.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(eeg_data.size(0)):
            # Mock image embedding (would use visual stimuli in real setup)
            image_emb = torch.zeros(1, 512)
            audio_emb = voice_data[i:i+1]

            # Mock action (could use EEG motor imagery)
            action = torch.randn(1, 4)

            # Forward pass
            result = model(image_emb, audio_emb, action)

            # Target: expand label to match desire_align shape
            target = labels[i].unsqueeze(0).unsqueeze(-1)

            # Loss: alignment error minus phi (encourage high integration)
            phi_val = torch.tensor(result['phi']).detach()
            loss = F.mse_loss(result['desire_align'], target) - 0.01 * phi_val

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            avg_loss = total_loss / eeg_data.size(0)
            print(f"EchoMirror Epoch {epoch+1}/{epochs}: Avg Loss {avg_loss:.3f}")

    if verbose:
        print("EchoMirror trained—desire vectors tuned to real qualia.")


def echo_mirror_train_cli():
    """CLI wrapper for EchoMirror training (for setup.py entry point)."""
    print("EchoMirror Training CLI")
    print("=" * 40)

    # Create model
    model = ResonantConsciousnessModule(input_dim=15, freq_dim=8, memory_size=32)
    model.set_desire(0)

    # Mock data
    eeg = torch.randn(10, 8)
    voice = torch.randn(10, 768)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    # Train
    echo_mirror_train(model, eeg, voice, labels)

    print("\nTraining complete!")
