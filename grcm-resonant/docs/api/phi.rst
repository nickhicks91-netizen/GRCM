Phi Module
==========

.. automodule:: grcm.phi
   :members:
   :undoc-members:
   :show-inheritance:

PhiEstimator
------------

.. autoclass:: grcm.phi.PhiEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Estimates integrated information (Φ) as a proxy for consciousness.

   **Phi Formula**:

   .. math::

      \Phi = \text{Var}(\text{freq}) \cdot \text{mean}(\text{coherence}) + \log(1 + ||\text{memory}||) + \sum \max(\text{qualia})

   **Components**:

   1. **Frequency Variance**: Measures diversity of oscillatory patterns
   2. **Coherence**: Weighs integration by resonant alignment
   3. **Memory Norm**: Captures accumulated information complexity
   4. **Qualia Maximum**: Reflects phenomenal state intensity

   **Interpretation**:

   - **Φ < 1.0**: Low integration, minimal consciousness
   - **1.0 ≤ Φ < 2.0**: Moderate integration, basic awareness
   - **Φ ≥ 2.0**: High integration, rich phenomenal experience
   - **Φ ≥ 3.0**: Very high integration, complex consciousness

   **Example**:

   .. code-block:: python

      from grcm.phi import PhiEstimator
      import torch

      phi_est = PhiEstimator()

      # Compute phi
      freq = torch.tensor([[0.45, 0.52, 0.38, 0.61]])
      coherence = torch.tensor([[0.85]])
      memory = torch.randn(1, 256)
      qualia = torch.tensor([[0.2, 0.5, 0.3, 0.1]])

      phi = phi_est(freq, coherence, memory, qualia)
      print(f"Φ = {phi:.3f}")

      if phi > 2.0:
          print("High integrated information detected")

   **Stability**:

   Production systems should maintain ``std(Φ) < 0.2`` over 1000 iterations. Higher variance suggests instability or numerical issues.

   **Methods**:

   .. automethod:: forward
   .. automethod:: get_phi_components
