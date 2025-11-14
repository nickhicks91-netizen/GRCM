Grounding Module
================

.. automodule:: grcm.grounding
   :members:
   :undoc-members:
   :show-inheritance:

MultimodalGrounding
-------------------

.. autoclass:: grcm.grounding.MultimodalGrounding
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Grounds multimodal perception by fusing vision, audio, and proprioceptive inputs.

   **Architecture**:

   1. **Image projection**: Maps CLIP embeddings to hidden_dim
   2. **Audio projection**: Maps Wav2Vec2 embeddings to hidden_dim
   3. **Proprioception projection**: Maps body state to hidden_dim
   4. **Fusion**: Concatenates and projects to unified representation

   .. code-block:: text

      [Image ViT] → Linear(512→256)
                                    ↘
      [Audio Wav2Vec2] → Linear(768→256) → Concat → Linear(256×3→256) → Grounded
                                    ↗
      [Proprioception] → Linear(8→256)

   **Example**:

   .. code-block:: python

      from grcm.grounding import MultimodalGrounding
      import torch

      grounding = MultimodalGrounding(
          image_dim=512,   # CLIP ViT-B/16
          audio_dim=768,   # Wav2Vec2-Base
          prop_dim=8,      # Body sensors
          hidden_dim=256
      )

      # Inputs from perception models
      image = torch.randn(1, 512)  # From CLIP
      audio = torch.randn(1, 768)  # From Wav2Vec2
      prop = torch.randn(1, 8)     # From proprioception sensors

      grounded = grounding(image, audio, prop)
      print(f"Grounded embedding: {grounded.shape}")  # [1, 256]

   **Recommended Encoders**:

   - **Vision**: CLIP ViT-B/16 (512-dim), CLIP ViT-L/14 (768-dim)
   - **Audio**: Wav2Vec2-Base (768-dim), Wav2Vec2-Large (1024-dim)
   - **Proprioception**: Custom sensor fusion (joint angles, IMU, touch)

   **Methods**:

   .. automethod:: forward
