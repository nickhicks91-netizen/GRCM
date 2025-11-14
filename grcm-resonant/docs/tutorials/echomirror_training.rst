EchoMirror Training Tutorial
============================

Train GRCM to mirror external signals (EEG, voice, biometrics) using EchoMirror.

What is EchoMirror?
-------------------

**EchoMirror** is GRCM's self-supervised training method that learns to reconstruct target frequency patterns from real biosignals:

- **EEG**: Theta/alpha band oscillations (4-12 Hz)
- **Voice**: Spectral patterns from voice recordings
- **Biometrics**: Heart rate variability, skin conductance

By minimizing reconstruction loss, GRCM's node frequencies and desire vectors adapt to match human-like patterns.

Setup
-----

.. code-block:: python

   import torch
   from grcm import ResonantConsciousnessModule
   from grcm.training import EchoMirror
   import mlflow

   # Initialize model
   model = ResonantConsciousnessModule(512, 768, 256)

   # Initialize trainer
   trainer = EchoMirror(
       model,
       lr=1e-3,
       freq_loss_weight=1.0
   )

Basic Training Loop
-------------------

.. code-block:: python

   # Mock training data (replace with real EEG/voice data)
   def generate_training_batch():
       image_emb = torch.randn(32, 512)
       audio_emb = torch.randn(32, 768)
       # Extract theta/alpha from EEG (4-12 Hz normalized to 0-1)
       freq_target = torch.rand(32, 1) * 0.3 + 0.2  # 0.2-0.5 range
       return image_emb, audio_emb, freq_target

   # Training loop
   num_epochs = 10
   mlflow.start_run()

   for epoch in range(num_epochs):
       epoch_losses = []

       for batch_idx in range(100):  # 100 batches per epoch
           image_emb, audio_emb, freq_target = generate_training_batch()

           # Train step
           loss = trainer.train_step(
               image_emb, audio_emb,
               freq_target=freq_target,
               desire_idx=0  # Train calm state
           )

           epoch_losses.append(loss)

       # Log to MLflow
       mean_loss = torch.tensor(epoch_losses).mean().item()
       mlflow.log_metric("loss", mean_loss, step=epoch)
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss:.4f}")

   mlflow.end_run()

**Expected Output**:

.. code-block:: text

   Epoch 1/10, Loss: 0.1234
   Epoch 2/10, Loss: 0.0987
   Epoch 3/10, Loss: 0.0823
   ...
   Epoch 10/10, Loss: 0.0321

Training Custom Desire States
------------------------------

.. code-block:: python

   # Train "flow" state (theta-alpha coherence)
   # This pattern is characteristic of deep focus

   # Load real EEG data showing theta-alpha coherence
   # (This is mock data - replace with real EEG)
   def get_flow_patterns():
       # Theta-alpha coherence typically shows frequency ~0.25-0.35
       return torch.rand(32, 1) * 0.1 + 0.25

   mlflow.start_run(run_name="train_flow_state")

   for epoch in range(20):
       for batch_idx in range(50):
           image_emb = torch.randn(32, 512)
           audio_emb = torch.randn(32, 768)
           flow_freq = get_flow_patterns()

           loss = trainer.train_desire_state(
               desire_idx=4,  # New desire state
               freq_target=flow_freq,
               image_emb=image_emb,
               audio_emb=audio_emb,
               num_epochs=1
           )

       print(f"Flow state training epoch {epoch+1}, Loss: {loss:.4f}")

   mlflow.end_run()

Using Real EEG Data
-------------------

.. code-block:: python

   import mne  # MNE-Python for EEG processing

   # Load EEG data (example with MNE)
   raw = mne.io.read_raw_fif('eeg_data.fif', preload=True)

   # Extract theta (4-8 Hz) and alpha (8-12 Hz) bands
   theta = raw.copy().filter(4, 8, fir_design='firwin')
   alpha = raw.copy().filter(8, 12, fir_design='firwin')

   # Compute band powers
   theta_power = theta.get_data().var(axis=1).mean()
   alpha_power = alpha.get_data().var(axis=1).mean()

   # Normalize to 0-1 frequency range
   theta_freq = theta_power / (theta_power + alpha_power) * 0.3 + 0.2
   alpha_freq = alpha_power / (theta_power + alpha_power) * 0.3 + 0.5

   # Use as training target
   freq_target = torch.tensor([[theta_freq]]).float()

   loss = trainer.train_step(
       image_emb, audio_emb,
       freq_target=freq_target,
       desire_idx=0
   )

Using Voice Data
----------------

.. code-block:: python

   import librosa

   # Load voice recording
   audio, sr = librosa.load('voice.wav', sr=16000)

   # Extract spectral centroid (brightness)
   spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

   # Normalize to 0-1 range
   freq_target = (spectral_centroid - spectral_centroid.min()) / \
                 (spectral_centroid.max() - spectral_centroid.min())

   # Average over time windows
   freq_target = torch.tensor(freq_target).mean().unsqueeze(0).unsqueeze(0)

   loss = trainer.train_step(
       image_emb, audio_emb,
       freq_target=freq_target,
       desire_idx=2  # Creative state
   )

MLflow Tracking
---------------

.. code-block:: python

   import mlflow
   from grcm.mlflow_logger import GRCMMLflowLogger

   # Setup MLflow
   mlflow.set_tracking_uri("http://localhost:5000")
   mlflow.set_experiment("echomirror_training")

   # Initialize logger
   logger = GRCMMLflowLogger("echomirror_calm")

   mlflow.start_run()

   # Log parameters
   mlflow.log_params({
       "learning_rate": 1e-3,
       "desire_state": 0,
       "num_epochs": 10,
       "batch_size": 32
   })

   # Training loop with logging
   for epoch in range(10):
       for step in range(100):
           image_emb, audio_emb, freq_target = generate_training_batch()

           loss = trainer.train_step(
               image_emb, audio_emb,
               freq_target=freq_target,
               desire_idx=0
           )

           # Forward pass for metrics
           result = model(image_emb[0:1], audio_emb[0:1], desire_idx=0)

           # Log to MLflow
           logger.log_step(result, step=epoch * 100 + step)
           logger.log_metric("loss", loss, step=epoch * 100 + step)

   mlflow.end_run()

   print("View results at: http://localhost:5000")

Validation
----------

.. code-block:: python

   # Validate on held-out data
   model.eval()

   with torch.no_grad():
       val_losses = []
       val_coherences = []

       for _ in range(50):
           image_emb, audio_emb, freq_target = generate_training_batch()

           result = model(image_emb[0:1], audio_emb[0:1], desire_idx=0)

           # Compute reconstruction loss
           freq_pred = model.embed(
               model.grounding(image_emb[0:1], audio_emb[0:1], model.body()),
               model.threads.identity_token
           )
           loss = torch.nn.functional.mse_loss(freq_pred, freq_target[0:1])

           val_losses.append(loss.item())
           val_coherences.append(result['coherence'].item())

   print(f"\nValidation Results:")
   print(f"  Loss: {torch.tensor(val_losses).mean():.4f}")
   print(f"  Coherence: {torch.tensor(val_coherences).mean():.3f}")
   print(f"  Coherence >0.7: {sum(1 for c in val_coherences if c > 0.7) / len(val_coherences) * 100:.1f}%")

Save Trained Model
------------------

.. code-block:: python

   # Save checkpoint
   trainer.save_checkpoint("checkpoints/echomirror_calm_epoch10.pth")

   # Register in MLflow
   mlflow.pytorch.log_model(model, "grcm_echomirror")

   print("Model saved and registered in MLflow")

Load and Use
------------

.. code-block:: python

   # Load trained model
   model_trained = ResonantConsciousnessModule(512, 768, 256)
   trainer_loaded = EchoMirror(model_trained)
   trainer_loaded.load_checkpoint("checkpoints/echomirror_calm_epoch10.pth")

   # Test inference
   result = model_trained(
       torch.randn(1, 512),
       torch.randn(1, 768),
       desire_idx=0
   )

   print(f"Trained model coherence: {result['coherence'].item():.3f}")
   print(f"Trained model phi: {result['phi']:.3f}")

Best Practices
--------------

1. **Use Real Biosignals**: Mock data is for testing only. Use real EEG/voice data for meaningful training.

2. **Normalize Frequencies**: Target frequencies should be in 0-1 range. Use domain knowledge to map biosignals appropriately.

3. **Monitor Coherence**: Aim for >70% of samples with coherence >0.7 after training.

4. **Phi Stability**: Check that phi std <0.2 over validation set.

5. **MLflow Tracking**: Always log experiments for reproducibility.

6. **Desire State Specificity**: Train each desire state with appropriate biosignal patterns (calm = theta, alert = beta, etc.).

Next Steps
----------

- :doc:`multimodal_integration` - Integrate with CLIP/Wav2Vec2
- :doc:`performance_optimization` - Optimize trained models
- See ``examples/mlflow_demo.py`` for complete training examples
