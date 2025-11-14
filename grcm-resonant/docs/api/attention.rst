Attention Module
================

.. automodule:: grcm.attention
   :members:
   :undoc-members:
   :show-inheritance:

ResonantAttention
-----------------

.. autoclass:: grcm.attention.ResonantAttention
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Frequency-based resonant attention mechanism with dynamic bandwidth modulation.

   **Coherence Formula**:

   .. math::

      \text{coherence} = \max\left(0, 1 - \frac{|\text{freq} - \text{node\_freq}|}{\text{bandwidth}}\right)

   where:

   - ``freq`` is the input frequency [batch, 1]
   - ``node_freq`` are learned node frequencies [num_nodes]
   - ``bandwidth`` is dynamically modulated base bandwidth
   - ``coherence`` is gated to [0, 1] range

   **Dynamic Bandwidth**:

   .. math::

      \text{bandwidth}_{\text{effective}} = \text{clamp}(\text{base\_bandwidth} + \text{bw\_bias}, 0.1, 1.0)

   ``bw_bias`` comes from desire alignment (0.2 × cosine similarity with desire vector).

   **Coherence Gating**:

   Only embeddings with coherence >0.7 (configurable threshold) pass through to downstream processing. This ensures only resonant, relevant information flows into memory and reflection.

   **Example**:

   .. code-block:: python

      from grcm.attention import ResonantAttention
      import torch

      attn = ResonantAttention(
          num_nodes=64,
          base_bandwidth=0.5
      )

      freq = torch.tensor([[0.35]])  # Frequency value
      bw_bias = torch.tensor([[0.15]])  # Desire-driven bias

      coherence = attn(freq, bw_bias)
      print(f"Coherence: {coherence.item():.3f}")

      # Check if passes threshold
      if coherence.item() > 0.7:
          print("Information passes coherence gating")
      else:
          print("Information filtered out")

   **Parameters**:

   - ``num_nodes`` (int): Number of resonant frequency nodes (default: 64)
   - ``base_bandwidth`` (float): Base bandwidth for coherence (default: 0.5)
   - ``init_freq_range`` (tuple): Range for initializing node frequencies (default: (0.0, 1.0))

   **Initialization**:

   Node frequencies are initialized uniformly across ``init_freq_range``. During training with EchoMirror, these frequencies adapt to match input distributions (EEG theta/alpha bands, voice spectra, etc.).

   **Methods**:

   .. automethod:: forward
   .. automethod:: get_coherence_stats
   .. automethod:: visualize_nodes
