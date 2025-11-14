Memory Module
=============

.. automodule:: grcm.memory
   :members:
   :undoc-members:
   :show-inheritance:

EpisodicMemory
--------------

.. autoclass:: grcm.memory.EpisodicMemory
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Episodic memory system using GRU for temporal integration.

   Integrates coherent information into persistent memory state via GRU cell:

   .. math::

      h_t = \text{GRU}(x_t, h_{t-1})

   Memory is only updated if coherence >0.7 (coherence gating).

   **Example**:

   .. code-block:: python

      from grcm.memory import EpisodicMemory
      import torch

      memory = EpisodicMemory(hidden_dim=256)

      # Initial memory state
      h = torch.zeros(1, 256)

      # Update with coherent input
      x = torch.randn(1, 256)
      coherence = torch.tensor([[0.85]])  # Above threshold

      h_new = memory(x, h, coherence)
      print(f"Memory updated: {not torch.equal(h, h_new)}")

   **Methods**:

   .. automethod:: forward
   .. automethod:: reset
   .. automethod:: get_memory_norm
