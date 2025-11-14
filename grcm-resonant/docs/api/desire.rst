Desire Module
=============

.. automodule:: grcm.desire
   :members:
   :undoc-members:
   :show-inheritance:

DesireSystem
------------

.. autoclass:: grcm.desire.DesireSystem
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Desire-driven agency system with pre-trained desire states and dynamic bandwidth modulation.

   **Desire States**:

   GRCM includes 4 pre-trained desire states, each with distinct attentional properties:

   +-------------+------------------+-------------------------+---------------------------+
   | Index       | State            | Bandwidth Bias          | Use Case                  |
   +=============+==================+=========================+===========================+
   | 0           | Calm             | -0.1 to 0.0 (narrow)    | Meditation, focused rest  |
   +-------------+------------------+-------------------------+---------------------------+
   | 1           | Alert            | +0.1 to +0.2 (broad)    | Vigilance, monitoring     |
   +-------------+------------------+-------------------------+---------------------------+
   | 2           | Creative         | +0.2 to +0.3 (diffuse)  | Exploration, ideation     |
   +-------------+------------------+-------------------------+---------------------------+
   | 3           | Focus            | -0.2 to -0.1 (sharp)    | Task completion, flow     |
   +-------------+------------------+-------------------------+---------------------------+

   **Desire Alignment**:

   Alignment is computed via cosine similarity between input frequency and the current desire vector:

   .. math::

      \text{alignment} = \cos(\text{freq}, \text{desire\_vec})

   **Bandwidth Bias**:

   .. math::

      \text{bw\_bias} = 0.2 \times \text{alignment}

   Higher alignment increases bandwidth (broader attention), lower alignment decreases it (narrower attention).

   **Desire Masking**:

   If alignment <0.5, the desire system considers the input misaligned and may suggest switching desire states. This prevents forcing incompatible desire states onto inputs.

   **Example**:

   .. code-block:: python

      from grcm.desire import DesireSystem
      import torch

      desire_sys = DesireSystem(
          hidden_dim=256,
          num_desires=4
      )

      freq = torch.randn(1, 256)
      desire_idx = 2  # Creative state

      desire_align, bw_bias = desire_sys(freq, desire_idx)

      print(f"Desire alignment: {desire_align.item():.3f}")
      print(f"Bandwidth bias: {bw_bias.item():.3f}")

      if desire_align.item() < 0.5:
          print("Warning: Low alignment, consider different desire state")

   **Custom Desire States**:

   You can train custom desire states using EchoMirror:

   .. code-block:: python

      from grcm.training import EchoMirror

      trainer = EchoMirror(model)

      # Train desire state 4 (custom: "flow")
      trainer.train_desire_state(
          desire_idx=4,
          eeg_data=eeg_theta_alpha,  # Target: theta/alpha coherence
          voice_data=voice_spectra,   # Target: calm voice patterns
          num_epochs=10
      )

   **Methods**:

   .. automethod:: forward
   .. automethod:: get_desire_vector
   .. automethod:: set_desire_vector
   .. automethod:: get_alignment_history
