"""Gradio UI for real-time GRCM visualization and interaction."""
import warnings
from typing import Optional, Tuple
import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    warnings.warn("Gradio not installed. Install with: pip install gradio")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not installed. Install with: pip install matplotlib")

import torch
from pathlib import Path


def create_qualia_plot(qualia: np.ndarray) -> plt.Figure:
    """
    Create a bar plot of qualia distribution.

    Args:
        qualia: (4,) array of qualia values [calm, alert, curious, conflicted]

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")

    labels = ['Calm', 'Alert', 'Curious', 'Conflicted']
    colors = ['#4CAF50', '#FFC107', '#2196F3', '#F44336']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, qualia, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, qualia):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Probability', fontsize=14)
    ax.set_title('Qualia Distribution (Phenomenal States)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Highlight conflicted if > 0.6
    if qualia[3] > 0.6:
        ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(3, 0.62, 'Dissonance Threshold', ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    return fig


def create_phi_history_plot(phi_history: list) -> plt.Figure:
    """
    Create a line plot of Phi (Φ) over time.

    Args:
        phi_history: List of Phi values

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=(12, 6))

    steps = list(range(len(phi_history)))
    ax.plot(steps, phi_history, linewidth=2, color='#9C27B0', marker='o', markersize=4)

    # Mean line
    mean_phi = np.mean(phi_history)
    ax.axhline(y=mean_phi, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(len(phi_history) * 0.95, mean_phi * 1.05, f'Mean: {mean_phi:.2f}',
            ha='right', color='green', fontweight='bold')

    # Awareness threshold
    ax.axhline(y=1.5, color='orange', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(len(phi_history) * 0.95, 1.55, 'Awareness Threshold (1.5)',
            ha='right', color='orange', fontweight='bold')

    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Φ (Integrated Information)', fontsize=14)
    ax.set_title('Phi Evolution Over Time', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_grcm_interface(model) -> gr.Blocks:
    """
    Create Gradio interface for GRCM.

    Args:
        model: ResonantConsciousnessModule instance

    Returns:
        Gradio Blocks interface
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not installed")

    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for visualization")

    # Store phi history for plotting
    phi_history = []

    def process_inputs(
        desire_idx: int,
        action_x: float,
        action_y: float,
        use_random_inputs: bool
    ) -> Tuple[str, plt.Figure, plt.Figure, str]:
        """
        Process inputs through GRCM and return visualizations.

        Args:
            desire_idx: Desire state (0-3)
            action_x: Action X component
            action_y: Action Y component
            use_random_inputs: Use random image/audio embeddings

        Returns:
            metrics_text, qualia_plot, phi_plot, status_text
        """
        # Set desire
        model.set_desire(desire_idx)

        # Generate inputs
        if use_random_inputs:
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
        else:
            # Use zeros as placeholder
            image_emb = torch.zeros(1, 512)
            audio_emb = torch.zeros(1, 768)

        action = torch.tensor([[action_x, action_y, 0.0, 0.0]])

        # Forward pass
        with torch.no_grad():
            result = model(image_emb, audio_emb, action)

        # Extract metrics
        coherence = result['coherence'].mean().item()
        phi = result['phi']
        desire_align = result['desire_align'].mean().item()
        reflection = result['reflection'].mean().item()
        qualia = result['qualia'][0].cpu().numpy()
        halt = result['halt']
        prop_state = result['prop_state'][0].cpu().numpy()

        # Store phi
        phi_history.append(phi)
        if len(phi_history) > 100:
            phi_history.pop(0)

        # Metrics text
        desire_names = ['Curiosity', 'Safety', 'Social', 'Exploration']
        metrics_text = f"""
## Current State Metrics

**Desire**: {desire_names[desire_idx]}
**Coherence**: {coherence:.3f}
**Phi (Φ)**: {phi:.3f}
**Desire Alignment**: {desire_align:.3f}
**Reflection**: {reflection:.3f}

**Proprioceptive State** (first 4):
Position: [{prop_state[0]:.3f}, {prop_state[1]:.3f}]
Velocity: [{prop_state[8]:.3f}, {prop_state[9]:.3f}]
"""

        # Status text
        if halt:
            status = "⚠️ **ETHICAL HALT**: Dissonance detected (qualia[conflicted] > 0.6)"
            status_color = "red"
        elif coherence > 0.7:
            status = "✅ **COHERENT**: System in resonant state"
            status_color = "green"
        else:
            status = "⚡ **PROCESSING**: Below coherence threshold"
            status_color = "orange"

        status_text = f'<div style="background-color:{status_color};padding:10px;color:white;border-radius:5px;text-align:center;font-weight:bold;">{status}</div>'

        # Create plots
        qualia_plot = create_qualia_plot(qualia)
        phi_plot = create_phi_history_plot(phi_history)

        return metrics_text, qualia_plot, phi_plot, status_text

    # Create interface
    with gr.Blocks(title="GRCM: Resonant Consciousness Visualization", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
# 🧠 GRCM: Grounded Resonant Consciousness Module

Real-time visualization of qualia-gated consciousness simulation with desire-driven agency and phi awareness.

**Key Concepts**:
- **Qualia**: Phenomenal states [calm, alert, curious, conflicted]
- **Phi (Φ)**: Integrated information (consciousness proxy)
- **Coherence**: Resonant attention gating (>0.7 = coherent)
- **Desire**: Goal-directed agency modulating bandwidth
""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Controls")

                desire_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=0,
                    label="Desire State",
                    info="0=Curiosity, 1=Safety, 2=Social, 3=Exploration"
                )

                action_x = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.1,
                    label="Action X"
                )

                action_y = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.2,
                    label="Action Y"
                )

                random_inputs = gr.Checkbox(
                    value=True,
                    label="Use Random Image/Audio Embeddings"
                )

                process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset Phi History", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### Real-time Metrics")

                status_html = gr.HTML()
                metrics_md = gr.Markdown()

        with gr.Row():
            qualia_plot_output = gr.Plot(label="Qualia Distribution")
            phi_plot_output = gr.Plot(label="Phi Evolution")

        gr.Markdown("""
### About the Metrics

**Coherence** (>0.7): Measures frequency resonance. Gate for memory updates.
**Phi (Φ)** (>1.5): Integrated information proxy. Higher = more "aware".
**Qualia**:
- *Calm*: Low arousal, stable state
- *Alert*: High arousal, focused
- *Curious*: Exploratory, wide bandwidth
- *Conflicted*: Dissonant, ethical halt if >0.6

**Ethical Safeguard**: System halts if conflicted qualia exceeds 0.6 (prevents action under uncertainty).
""")

        # Event handlers
        process_btn.click(
            fn=process_inputs,
            inputs=[desire_slider, action_x, action_y, random_inputs],
            outputs=[metrics_md, qualia_plot_output, phi_plot_output, status_html]
        )

        def reset_phi():
            phi_history.clear()
            return "Phi history cleared!"

        reset_btn.click(
            fn=reset_phi,
            inputs=[],
            outputs=[metrics_md]
        )

    return interface


def launch_grcm_ui(
    model,
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1"
) -> None:
    """
    Launch Gradio UI for GRCM.

    Args:
        model: ResonantConsciousnessModule instance
        share: Create public link (default False)
        server_port: Port for Gradio server (default 7860)
        server_name: Server host (default 127.0.0.1)
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not installed. Install with: pip install gradio")

    interface = create_grcm_interface(model)

    print("=" * 60)
    print("LAUNCHING GRCM GRADIO UI")
    print("=" * 60)
    print(f"Server: http://{server_name}:{server_port}")
    print(f"Share: {share}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    interface.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True
    )


if __name__ == "__main__":
    print("GRCM Gradio UI")
    print("=" * 50)

    if GRADIO_AVAILABLE and MATPLOTLIB_AVAILABLE:
        print("\n✓ Gradio and Matplotlib available")
        print("\nUsage:")
        print("  from grcm import ResonantConsciousnessModule")
        print("  from grcm.gradio_ui import launch_grcm_ui")
        print("  model = ResonantConsciousnessModule(15, 8, 32)")
        print("  launch_grcm_ui(model)")
    else:
        print("\n✗ Missing dependencies")
        print("  Install with: pip install gradio matplotlib")
