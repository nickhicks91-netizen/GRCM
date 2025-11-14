Core Module
===========

.. automodule:: grcm.core
   :members:
   :undoc-members:
   :show-inheritance:

ResonantConsciousnessModule
---------------------------

.. autoclass:: grcm.core.ResonantConsciousnessModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Main orchestration module that coordinates all GRCM components.

   **Architecture**:

   1. **Grounding**: Fuses multimodal inputs (image + audio + proprioception)
   2. **Embedding**: Maps to frequency space with identity token influence
   3. **Desire Alignment**: Computes alignment with current desire state
   4. **Resonant Attention**: Gates information via coherence threshold (>0.7)
   5. **Memory Update**: Integrates into episodic memory via GRU
   6. **Reflection**: Processes memory to generate conscious output
   7. **Qualia Generation**: Simulates phenomenal states
   8. **Phi Estimation**: Approximates integrated information
   9. **Ethical Check**: Halts if dissonance >0.6

   **Example**:

   .. code-block:: python

      import torch
      from grcm import ResonantConsciousnessModule

      model = ResonantConsciousnessModule(
          image_dim=512,
          audio_dim=768,
          hidden_dim=256,
          num_desires=4,
          num_nodes=64,
          num_qualia=4
      )

      # Forward pass
      result = model(
          image_emb=torch.randn(1, 512),
          audio_emb=torch.randn(1, 768),
          action=torch.randn(1, 4),
          desire_idx=0
      )

      print(f"Coherence: {result['coherence'].item():.3f}")
      print(f"Phi: {result['phi']:.3f}")
      print(f"Halt: {result['halt']}")

   **Return Dictionary**:

   The forward method returns a dictionary with the following keys:

   - ``output`` (Tensor): Reflected conscious output [batch, hidden_dim]
   - ``coherence`` (Tensor): Mean coherence score [batch, 1]
   - ``memory`` (Tensor): Updated memory state [batch, hidden_dim]
   - ``qualia`` (Tensor): Phenomenal state distribution [batch, num_qualia]
   - ``phi`` (float): Integrated information estimate
   - ``prop_state`` (Tensor): Proprioceptive body state [batch, prop_dim]
   - ``desire_align`` (Tensor): Alignment with current desire [batch, 1]
   - ``reflection`` (Tensor): Reflective output [batch, hidden_dim]
   - ``halt`` (bool): Ethical halt flag (True if dissonance >0.6)

   **Configuration**:

   All parameters can be configured via YAML:

   .. code-block:: yaml

      model:
        image_dim: 512
        audio_dim: 768
        hidden_dim: 256
        num_desires: 4
        num_nodes: 64
        num_qualia: 4
        prop_dim: 8

      coherence:
        threshold: 0.7
        base_bandwidth: 0.5

      phi:
        threshold: 1.5

      ethics:
        dissonance_threshold: 0.6

   **Methods**:

   .. automethod:: forward
   .. automethod:: reset_memory
   .. automethod:: set_desire_state
   .. automethod:: get_phi_history
   .. automethod:: load_config
   .. automethod:: save_checkpoint
   .. automethod:: load_checkpoint
