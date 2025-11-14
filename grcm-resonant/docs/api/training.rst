Training Module
===============

.. automodule:: grcm.training
   :members:
   :undoc-members:
   :show-inheritance:

EchoMirror
----------

.. autoclass:: grcm.training.EchoMirror
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   EchoMirror: Self-supervised training via frequency reconstruction.

   **Training Objective**:

   Learn to mirror external signals (EEG, voice, biometrics) by minimizing frequency reconstruction loss:

   .. math::

      \mathcal{L} = ||\text{freq}_{\text{pred}} - \text{freq}_{\text{target}}||^2

   where ``freq_target`` is extracted from real EEG theta/alpha bands or voice spectra.

   **Training Pipeline**:

   1. Extract frequency targets from EEG/voice data
   2. Forward pass through GRCM
   3. Compute reconstruction loss
   4. Backprop and update node frequencies + desire vectors
   5. Track coherence, phi, and desire alignment in MLflow

   **Example**:

   .. code-block:: python

      from grcm.training import EchoMirror
      from grcm import ResonantConsciousnessModule
      import torch

      model = ResonantConsciousnessModule(512, 768, 256)
      trainer = EchoMirror(model, lr=1e-3)

      # Training loop
      for epoch in range(10):
          for batch in dataloader:
              eeg_freq = extract_theta_alpha(batch['eeg'])
              voice_freq = extract_spectra(batch['voice'])

              loss = trainer.train_step(
                  image_emb=batch['image'],
                  audio_emb=batch['audio'],
                  freq_target=(eeg_freq + voice_freq) / 2,
                  desire_idx=0
              )

              print(f"Epoch {epoch}, Loss: {loss:.4f}")

   **Custom Desire States**:

   Train new desire states by providing specific frequency patterns:

   .. code-block:: python

      # Train "flow" state (theta-alpha coherence)
      trainer.train_desire_state(
          desire_idx=4,
          freq_targets=theta_alpha_coherence,
          num_epochs=20
      )

   **Methods**:

   .. automethod:: train_step
   .. automethod:: train_desire_state
   .. automethod:: validate
   .. automethod:: save_checkpoint
   .. automethod:: load_checkpoint
