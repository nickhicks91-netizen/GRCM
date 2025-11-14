GRCM-Resonant: Grounded Resonant Consciousness Module
=====================================================

**GRCM-Resonant** is a production-ready implementation of a resonant consciousness simulation framework that grounds multimodal perception (vision + audio) through frequency-based coherence gating and desire-driven agency.

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-2.1%2B-red
   :target: https://pytorch.org/
   :alt: PyTorch Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/nickhicks91-netizen/GRCM/blob/main/LICENSE
   :alt: License

Key Features
------------

✨ **Resonant Attention**
   Frequency-based coherence gating with dynamic bandwidth modulation:

   .. math::

      \text{coherence} = \max(0, 1 - \frac{|\text{freq} - \text{node\_freq}|}{\text{bandwidth}})

🎯 **Desire-Driven Agency**
   Four pre-trained desire states (calm, alert, creative, focus) with desire masking (>0.5 alignment threshold)

🧠 **Integrated Information Proxy (Φ)**
   Approximates consciousness with phi estimation:

   .. math::

      \Phi = \text{Var}(\text{freq}) \cdot \text{mean}(\text{coherence}) + \log(1 + ||\text{memory}||) + \sum \max(\text{qualia})

🌈 **Qualia Simulation**
   Four phenomenal states mapped to frequency bands (calm, alert, creative, dissonance)

⚖️ **Ethical Safeguards**
   Automatic halt when dissonance (qualia[3]) exceeds 0.6 threshold

🔄 **Episodic Threading**
   Temporal coherence tracking across sequences with identity token persistence

🚀 **Production Ready**
   Docker, Kubernetes, BentoML API, Prometheus monitoring, CI/CD pipelines

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install grcm-resonant

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from grcm import ResonantConsciousnessModule

   # Initialize GRCM
   model = ResonantConsciousnessModule(
       image_dim=512,      # CLIP ViT-B/16 embedding
       audio_dim=768,      # Wav2Vec2 embedding
       hidden_dim=256
   )

   # Forward pass
   image_emb = torch.randn(1, 512)  # From CLIP
   audio_emb = torch.randn(1, 768)  # From Wav2Vec2

   result = model(image_emb, audio_emb, desire_idx=0)  # 0 = calm

   # Access outputs
   print(f"Coherence: {result['coherence'].item():.3f}")
   print(f"Phi (Φ): {result['phi']:.3f}")
   print(f"Qualia: {result['qualia']}")
   print(f"Ethical Halt: {result['halt']}")

Docker Quick Start
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start full stack (API + MLflow + Gradio + Monitoring)
   docker-compose up -d

   # Access services
   # API: http://localhost:3000
   # MLflow: http://localhost:5000
   # Gradio UI: http://localhost:7860
   # Prometheus: http://localhost:9091
   # Grafana: http://localhost:3001 (admin/admin)

   # Make prediction
   curl -X POST http://localhost:3000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "image_emb": [...],
       "audio_emb": [...],
       "desire_idx": 0
     }'

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/quickstart
   guides/configuration
   guides/deployment

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_usage
   tutorials/echomirror_training
   tutorials/multimodal_integration
   tutorials/performance_optimization

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/core
   api/grounding
   api/attention
   api/desire
   api/memory
   api/qualia
   api/phi
   api/training
   api/optimization
   api/serving

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview
   architecture/coherence_gating
   architecture/desire_system
   architecture/phi_estimation
   architecture/ethical_safeguards

.. toctree::
   :maxdepth: 1
   :caption: Deployment

   deployment/docker
   deployment/kubernetes
   deployment/monitoring
   deployment/runbooks

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Performance
-----------

Optimization Results
~~~~~~~~~~~~~~~~~~~~

+-----------------+---------------+------------------+---------------------+
| Configuration   | Latency (ms)  | Throughput       | Coherence Quality   |
+=================+===============+==================+=====================+
| Baseline        | 122.4 ± 8.3   | 8.2 it/s         | 78.5% > 0.7         |
+-----------------+---------------+------------------+---------------------+
| torch.compile   | 48.7 ± 3.1    | 20.5 it/s (2.5x) | 79.1% > 0.7         |
+-----------------+---------------+------------------+---------------------+
| INT8 Quant      | 67.2 ± 4.8    | 14.9 it/s (1.8x) | 78.9% > 0.7         |
+-----------------+---------------+------------------+---------------------+
| TensorRT FP16   | 8.4 ± 0.6     | 119 it/s (14.5x) | 79.3% > 0.7         |
+-----------------+---------------+------------------+---------------------+

Production Targets
~~~~~~~~~~~~~~~~~~

- **Latency**: <50ms p95 on CPU, <10ms on GPU
- **Throughput**: >100 req/s sustained
- **Coherence**: >95% samples with coherence >0.7
- **Phi Stability**: std <0.2 over 1000 iterations
- **Availability**: 99.9% uptime

Key Concepts
------------

Resonant Coherence
~~~~~~~~~~~~~~~~~~

GRCM uses frequency-based coherence to gate information flow. Each embedding is mapped to a frequency value, and coherence is computed based on proximity to learned node frequencies:

.. math::

   \text{coherence} = \max\left(0, 1 - \frac{|\text{freq} - \text{node\_freq}|}{\text{bandwidth}}\right)

Only information with coherence >0.7 passes through to downstream processing.

Desire-Driven Modulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Four pre-trained desire states modulate bandwidth dynamically:

- **Calm (0)**: Narrow, focused attention (bandwidth ≈ 0.4-0.5)
- **Alert (1)**: Broad, vigilant attention (bandwidth ≈ 0.6-0.7)
- **Creative (2)**: Diffuse, exploratory attention (bandwidth ≈ 0.7-0.8)
- **Focus (3)**: Laser-sharp attention (bandwidth ≈ 0.3-0.4)

Desire alignment is computed via cosine similarity, and bandwidth is biased by 0.2 × alignment.

Phi Estimation (IIT Proxy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An approximation of Integrated Information Theory's Φ:

.. math::

   \Phi = \text{Var}(\text{freq}) \cdot \text{mean}(\text{coherence}) + \log(1 + ||\text{memory}||) + \sum \max(\text{qualia})

Higher Φ suggests richer phenomenal experience and integrated information processing.

Ethical Safeguards
~~~~~~~~~~~~~~~~~~

When dissonance (qualia[3]) exceeds 0.6, GRCM halts processing and returns None for predictions. This prevents the system from acting under conditions of internal conflict or uncertainty.

Citations
---------

If you use GRCM in your research, please cite:

.. code-block:: bibtex

   @software{grcm_resonant_2025,
     title={GRCM-Resonant: Grounded Resonant Consciousness Module},
     author={GRCM Development Team},
     year={2025},
     url={https://github.com/nickhicks91-netizen/GRCM}
   }

Related Work
~~~~~~~~~~~~

- **Integrated Information Theory (IIT)**: Tononi et al. (2016)
- **Global Workspace Theory (GWT)**: Baars (1988)
- **Predictive Processing**: Friston (2010)
- **Attention Schema Theory**: Graziano (2013)

Community
---------

- **GitHub**: https://github.com/nickhicks91-netizen/GRCM
- **Issues**: https://github.com/nickhicks91-netizen/GRCM/issues
- **Discussions**: https://github.com/nickhicks91-netizen/GRCM/discussions

License
-------

GRCM-Resonant is released under the MIT License. See :doc:`license` for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
