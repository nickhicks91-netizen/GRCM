#!/usr/bin/env python3
"""
Gradio UI Demo for GRCM

Launches interactive web interface for real-time GRCM visualization.

Features:
    - Real-time qualia visualization
    - Phi evolution tracking
    - Desire state control
    - Action input
    - Ethical halt monitoring

Usage:
    python examples/gradio_demo.py
    # Opens browser at http://127.0.0.1:7860
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from grcm import ResonantConsciousnessModule
    from grcm.gradio_ui import launch_grcm_ui
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


def main():
    """Launch Gradio UI."""
    if not IMPORTS_OK:
        print("✗ Import error. Install dependencies:")
        print("  pip install gradio matplotlib")
        return 1

    print("=" * 70)
    print("GRCM GRADIO UI DEMO")
    print("=" * 70)
    print("\nCreating GRCM model...")

    # Create model
    model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )
    model.eval()

    print("✓ Model created")
    print("\nLaunching Gradio UI...")
    print("  - Qualia visualization (bar chart)")
    print("  - Phi evolution (line plot)")
    print("  - Desire state control (slider)")
    print("  - Action input (sliders)")
    print("  - Real-time metrics display")

    # Launch UI
    try:
        launch_grcm_ui(
            model,
            share=False,  # Set to True for public link
            server_port=7860,
            server_name="127.0.0.1"
        )
    except KeyboardInterrupt:
        print("\n\n✓ Gradio UI stopped")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
