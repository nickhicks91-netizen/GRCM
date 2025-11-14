Quick Start Guide
=================

Get up and running with GRCM in 5 minutes.

Installation
------------

.. code-block:: bash

   pip install grcm-resonant

Basic Usage
-----------

**Minimal Example**:

.. code-block:: python

   import torch
   from grcm import ResonantConsciousnessModule

   # Initialize GRCM
   model = ResonantConsciousnessModule(
       image_dim=512,      # CLIP ViT-B/16
       audio_dim=768,      # Wav2Vec2-Base
       hidden_dim=256
   )

   # Create random embeddings (replace with real CLIP/Wav2Vec2)
   image_emb = torch.randn(1, 512)
   audio_emb = torch.randn(1, 768)

   # Forward pass
   result = model(image_emb, audio_emb, desire_idx=0)

   # Access outputs
   print(f"Coherence: {result['coherence'].item():.3f}")
   print(f"Phi (Φ): {result['phi']:.3f}")
   print(f"Qualia: {result['qualia'].tolist()}")
   print(f"Ethical halt: {result['halt']}")

**Expected Output**:

.. code-block:: text

   Coherence: 0.847
   Phi (Φ): 2.134
   Qualia: [0.45, 0.32, 0.18, 0.05]
   Ethical halt: False

Using Real Encoders
-------------------

**CLIP for Vision**:

.. code-block:: python

   from transformers import CLIPProcessor, CLIPModel
   from PIL import Image

   # Load CLIP
   clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
   clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

   # Encode image
   image = Image.open("photo.jpg")
   inputs = clip_processor(images=image, return_tensors="pt")
   image_emb = clip_model.get_image_features(**inputs)  # [1, 512]

**Wav2Vec2 for Audio**:

.. code-block:: python

   from transformers import Wav2Vec2Processor, Wav2Vec2Model
   import torchaudio

   # Load Wav2Vec2
   wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
   wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

   # Encode audio
   waveform, sr = torchaudio.load("audio.wav")
   inputs = wav2vec_processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
   audio_emb = wav2vec_model(**inputs).last_hidden_state.mean(dim=1)  # [1, 768]

**Complete Pipeline**:

.. code-block:: python

   from grcm import ResonantConsciousnessModule

   model = ResonantConsciousnessModule(512, 768, 256)

   # Process multimodal input
   result = model(image_emb, audio_emb, desire_idx=0)

   if result['halt']:
       print("WARNING: Ethical halt triggered (high dissonance)")
   else:
       print(f"Conscious output: {result['output'].shape}")

Desire States
-------------

GRCM has 4 pre-trained desire states:

.. code-block:: python

   # 0: Calm (narrow, focused)
   result_calm = model(image_emb, audio_emb, desire_idx=0)

   # 1: Alert (broad, vigilant)
   result_alert = model(image_emb, audio_emb, desire_idx=1)

   # 2: Creative (diffuse, exploratory)
   result_creative = model(image_emb, audio_emb, desire_idx=2)

   # 3: Focus (laser-sharp)
   result_focus = model(image_emb, audio_emb, desire_idx=3)

   # Compare coherence across desires
   print(f"Calm coherence: {result_calm['coherence'].item():.3f}")
   print(f"Alert coherence: {result_alert['coherence'].item():.3f}")
   print(f"Creative coherence: {result_creative['coherence'].item():.3f}")
   print(f"Focus coherence: {result_focus['coherence'].item():.3f}")

Sequential Processing
---------------------

GRCM maintains episodic memory across sequences:

.. code-block:: python

   model = ResonantConsciousnessModule(512, 768, 256)

   # Process video frames sequentially
   for frame in video_frames:
       image_emb = clip_model.get_image_features(frame)
       audio_emb = wav2vec_model(audio_chunk).last_hidden_state.mean(dim=1)

       result = model(image_emb, audio_emb, desire_idx=0)

       print(f"Frame {i}: Phi = {result['phi']:.3f}, Coherence = {result['coherence'].item():.3f}")

   # Memory accumulates across frames
   # Phi should increase as information integrates

Ethical Safeguards
------------------

GRCM automatically halts when dissonance is high:

.. code-block:: python

   result = model(image_emb, audio_emb)

   if result['halt']:
       print("Ethical halt triggered!")
       print(f"Dissonance: {result['qualia'][0, 3].item():.3f}")
       print("Suggestion: Review input or switch desire state")
   else:
       # Safe to use output
       output = result['output']

Configuration
-------------

Use YAML for configuration:

.. code-block:: yaml

   # config.yaml
   model:
     image_dim: 512
     audio_dim: 768
     hidden_dim: 256

   coherence:
     threshold: 0.7
     base_bandwidth: 0.5

   phi:
     threshold: 1.5

   ethics:
     dissonance_threshold: 0.6

.. code-block:: python

   from grcm import ResonantConsciousnessModule

   model = ResonantConsciousnessModule.from_config("config.yaml")

Docker Quick Start
------------------

.. code-block:: bash

   # Start full stack
   docker-compose up -d

   # Make API request
   curl -X POST http://localhost:3000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "image_emb": [...],  # 512-dim array
       "audio_emb": [...],  # 768-dim array
       "desire_idx": 0
     }'

   # View Gradio UI
   # Open http://localhost:7860 in browser

   # View MLflow experiments
   # Open http://localhost:5000 in browser

   # View Grafana dashboards
   # Open http://localhost:3001 (admin/admin)

Next Steps
----------

**Tutorials**:

- :doc:`../tutorials/basic_usage` - Detailed walkthrough
- :doc:`../tutorials/echomirror_training` - Train custom desire states
- :doc:`../tutorials/multimodal_integration` - Integrate with CLIP/Wav2Vec2
- :doc:`../tutorials/performance_optimization` - Optimize for production

**Deployment**:

- :doc:`../deployment/docker` - Docker deployment
- :doc:`../deployment/kubernetes` - Kubernetes orchestration
- :doc:`../deployment/monitoring` - Prometheus + Grafana

**API Reference**:

- :doc:`../api/core` - Core module documentation
- :doc:`../api/attention` - Resonant attention mechanism
- :doc:`../api/desire` - Desire system
