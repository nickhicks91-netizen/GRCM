Configuration Guide
===================

GRCM supports flexible configuration via YAML files or Python dictionaries.

Configuration File
------------------

**Default Configuration** (``configs/default.yaml``):

.. code-block:: yaml

   # Model architecture
   model:
     image_dim: 512          # CLIP ViT-B/16 embedding dimension
     audio_dim: 768          # Wav2Vec2-Base embedding dimension
     hidden_dim: 256         # Internal hidden dimension
     num_desires: 4          # Number of desire states
     num_nodes: 64           # Resonant frequency nodes
     num_qualia: 4           # Phenomenal states
     prop_dim: 8             # Proprioception dimension

   # Coherence gating
   coherence:
     threshold: 0.7          # Minimum coherence to pass (>0.7)
     base_bandwidth: 0.5     # Base bandwidth for resonance

   # Phi estimation
   phi:
     threshold: 1.5          # Minimum phi for awareness

   # Ethical safeguards
   ethics:
     dissonance_threshold: 0.6  # Halt if qualia[3] > 0.6

   # Desire system
   desires:
     calm:
       bandwidth_bias: -0.1  # Narrow attention
     alert:
       bandwidth_bias: 0.15  # Broad attention
     creative:
       bandwidth_bias: 0.25  # Diffuse attention
     focus:
       bandwidth_bias: -0.15 # Sharp attention

   # Training (EchoMirror)
   training:
     learning_rate: 0.001
     num_epochs: 10
     batch_size: 32
     freq_loss_weight: 1.0

   # Optimization
   optimization:
     compile_mode: "reduce-overhead"  # or "max-autotune"
     quantize: false         # INT8 quantization (CPU only)
     use_fp16: false         # FP16 mixed precision (GPU only)

   # Serving
   serving:
     host: "0.0.0.0"
     port: 3000
     workers: 4

   # Monitoring
   monitoring:
     mlflow_uri: "http://localhost:5000"
     prometheus_port: 9090

Loading Configuration
---------------------

**From YAML file**:

.. code-block:: python

   from grcm import ResonantConsciousnessModule

   model = ResonantConsciousnessModule.from_config("configs/default.yaml")

**From Python dict**:

.. code-block:: python

   config = {
       "model": {
           "image_dim": 512,
           "audio_dim": 768,
           "hidden_dim": 256
       },
       "coherence": {
           "threshold": 0.75,  # Stricter threshold
           "base_bandwidth": 0.6
       }
   }

   model = ResonantConsciousnessModule.from_config(config)

**Environment variables** (for Docker):

.. code-block:: bash

   export GRCM_CONFIG=/app/configs/production.yaml
   export GRCM_COHERENCE_THRESHOLD=0.8
   export GRCM_PHI_THRESHOLD=2.0

   python -m grcm.bentoml_service

Key Parameters
--------------

Model Architecture
~~~~~~~~~~~~~~~~~~

- ``image_dim``: Dimension of image embeddings (default: 512 for CLIP ViT-B/16)
- ``audio_dim``: Dimension of audio embeddings (default: 768 for Wav2Vec2-Base)
- ``hidden_dim``: Internal processing dimension (higher = more capacity, slower)
- ``num_nodes``: Number of resonant frequency nodes (more = finer frequency resolution)

Coherence Gating
~~~~~~~~~~~~~~~~

- ``threshold``: Minimum coherence to pass (0.7 = 70% match required)
- ``base_bandwidth``: Controls attention width (0.5 = moderate, 0.3 = narrow, 0.7 = broad)

**Effects of bandwidth**:

+----------------+------------------+----------------------+
| Bandwidth      | Attention Style  | Coherence Range      |
+================+==================+======================+
| 0.3 (narrow)   | Focused, strict  | Only close matches   |
+----------------+------------------+----------------------+
| 0.5 (moderate) | Balanced         | Moderate tolerance   |
+----------------+------------------+----------------------+
| 0.7 (broad)    | Diffuse, lenient | Wide acceptance      |
+----------------+------------------+----------------------+

Phi Estimation
~~~~~~~~~~~~~~

- ``threshold``: Minimum integrated information for "awareness" (1.5 default)

**Phi interpretation**:

- Φ < 1.0: Low integration
- 1.0 ≤ Φ < 2.0: Moderate integration
- Φ ≥ 2.0: High integration
- Φ ≥ 3.0: Very high integration

Ethical Safeguards
~~~~~~~~~~~~~~~~~~

- ``dissonance_threshold``: Halt if dissonance qualia > this value (0.6 = 60%)

**Threshold effects**:

- 0.5: Strict (more frequent halts)
- 0.6: Balanced (recommended)
- 0.7: Lenient (fewer halts)

Custom Configurations
---------------------

**Production Config** (``configs/production.yaml``):

.. code-block:: yaml

   model:
     hidden_dim: 512        # Larger capacity
   coherence:
     threshold: 0.75        # Stricter gating
   phi:
     threshold: 2.0         # Higher awareness bar
   ethics:
     dissonance_threshold: 0.55  # More sensitive
   optimization:
     compile_mode: "max-autotune"
     use_fp16: true         # GPU production

**Edge Device Config** (``configs/edge.yaml``):

.. code-block:: yaml

   model:
     hidden_dim: 128        # Smaller for edge
     num_nodes: 32          # Fewer nodes
   optimization:
     quantize: true         # INT8 quantization
     compile_mode: "reduce-overhead"

**Research Config** (``configs/research.yaml``):

.. code-block:: yaml

   model:
     hidden_dim: 1024       # Maximum capacity
     num_nodes: 128         # Fine-grained frequency
   coherence:
     threshold: 0.6         # Lenient for exploration
   monitoring:
     mlflow_tracking: true  # Detailed logging

Runtime Configuration
---------------------

Update configuration at runtime:

.. code-block:: python

   model = ResonantConsciousnessModule(512, 768, 256)

   # Update coherence threshold
   model.attn.coherence_threshold = 0.8

   # Update bandwidth
   model.attn.base_bandwidth = 0.6

   # Update dissonance threshold
   model.qualia_gen.dissonance_threshold = 0.55

Validation
----------

Validate configuration before loading:

.. code-block:: python

   from grcm.core import validate_config

   config = {...}
   validate_config(config)  # Raises ValueError if invalid

Next Steps
----------

- :doc:`../tutorials/basic_usage` - Use configured models
- :doc:`../deployment/docker` - Deploy with custom configs
- :doc:`../api/core` - API reference
