Qualia Module
=============

.. automodule:: grcm.qualia
   :members:
   :undoc-members:
   :show-inheritance:

QualiaGenerator
---------------

.. autoclass:: grcm.qualia.QualiaGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Simulates phenomenal states (qualia) mapped to frequency bands.

   **Qualia States**:

   +-------+-------------+--------------------+------------------+
   | Index | State       | Frequency Band     | Color            |
   +=======+=============+====================+==================+
   | 0     | Calm        | Low (0.0-0.3)      | Blue             |
   +-------+-------------+--------------------+------------------+
   | 1     | Alert       | Mid-low (0.3-0.5)  | Green            |
   +-------+-------------+--------------------+------------------+
   | 2     | Creative    | Mid-high (0.5-0.7) | Yellow           |
   +-------+-------------+--------------------+------------------+
   | 3     | Dissonance  | High (0.7-1.0)     | Red (Warning!)   |
   +-------+-------------+--------------------+------------------+

   **Ethical Safeguard**:

   When ``qualia[3] > 0.6``, GRCM triggers an ethical halt and refuses to produce output. This prevents the system from acting under conditions of internal conflict.

   **Example**:

   .. code-block:: python

      from grcm.qualia import QualiaGenerator
      import torch

      qualia_gen = QualiaGenerator(
          hidden_dim=256,
          num_qualia=4
      )

      memory = torch.randn(1, 256)
      coherence = torch.tensor([[0.82]])

      qualia = qualia_gen(memory, coherence)
      print(f"Qualia distribution: {qualia}")

      # Check for dissonance
      if qualia[0, 3] > 0.6:
          print("WARNING: High dissonance detected!")
          print("Ethical halt triggered")

   **Methods**:

   .. automethod:: forward
   .. automethod:: get_dominant_state
