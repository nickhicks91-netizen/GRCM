Installation Guide
==================

GRCM-Resonant supports multiple installation methods for different use cases.

Requirements
------------

**System Requirements**:

- Python 3.10 or higher
- PyTorch 2.1.0 or higher
- 4GB RAM minimum (8GB+ recommended)
- GPU optional (CUDA 11.8+ or ROCm 5.7+ for GPU acceleration)

**Operating Systems**:

- Linux (Ubuntu 20.04+, CentOS 8+, Debian 11+)
- macOS 12.0+ (Intel or Apple Silicon)
- Windows 10/11 with WSL2 (recommended) or native Python

Basic Installation
------------------

**Via PyPI** (recommended):

.. code-block:: bash

   pip install grcm-resonant

**From Source**:

.. code-block:: bash

   git clone https://github.com/nickhicks91-netizen/GRCM.git
   cd GRCM/grcm-resonant
   pip install -e .

**Verify Installation**:

.. code-block:: python

   import torch
   from grcm import ResonantConsciousnessModule

   model = ResonantConsciousnessModule(512, 768, 256)
   print("GRCM installed successfully!")

Optional Dependencies
---------------------

**Optimization** (torch.compile, ONNX, quantization):

.. code-block:: bash

   pip install grcm-resonant[optimization]

   # Includes: onnx, onnxruntime, onnx-simplifier

**Serving** (BentoML REST API):

.. code-block:: bash

   pip install grcm-resonant[serving]

   # Includes: bentoml, pydantic

**Monitoring** (Prometheus, MLflow, Gradio):

.. code-block:: bash

   pip install grcm-resonant[monitoring]

   # Includes: mlflow, gradio, prometheus-client

**Testing** (pytest, coverage):

.. code-block:: bash

   pip install grcm-resonant[testing]

   # Includes: pytest, pytest-cov, pytest-benchmark

**All Dependencies**:

.. code-block:: bash

   pip install grcm-resonant[all]

   # Includes: optimization, serving, monitoring, testing

GPU Support
-----------

**NVIDIA GPUs** (CUDA):

.. code-block:: bash

   # Install PyTorch with CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install GRCM
   pip install grcm-resonant

   # Verify GPU
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

**AMD GPUs** (ROCm):

.. code-block:: bash

   # Install PyTorch with ROCm 5.7
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

   # Install GRCM
   pip install grcm-resonant

**Apple Silicon** (MPS):

.. code-block:: bash

   # PyTorch with MPS support (macOS 12.3+)
   pip install torch torchvision torchaudio

   # Install GRCM
   pip install grcm-resonant

   # Verify MPS
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

Docker Installation
-------------------

**Pull Pre-built Image**:

.. code-block:: bash

   docker pull ghcr.io/nickhicks91-netizen/grcm:latest

   # Run container
   docker run -p 3000:3000 ghcr.io/nickhicks91-netizen/grcm:latest

**Build from Source**:

.. code-block:: bash

   git clone https://github.com/nickhicks91-netizen/GRCM.git
   cd GRCM/grcm-resonant
   docker build -t grcm-resonant .

**Docker Compose** (full stack):

.. code-block:: bash

   docker-compose up -d

   # Access services:
   # - API: http://localhost:3000
   # - MLflow: http://localhost:5000
   # - Gradio UI: http://localhost:7860
   # - Prometheus: http://localhost:9091
   # - Grafana: http://localhost:3001

Development Installation
-------------------------

For contributors and developers:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/nickhicks91-netizen/GRCM.git
   cd GRCM/grcm-resonant

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in editable mode with all dependencies
   pip install -e .[all]

   # Install pre-commit hooks
   pip install pre-commit
   pre-commit install

   # Run tests
   pytest

Troubleshooting
---------------

**ImportError: No module named 'grcm'**:

.. code-block:: bash

   # Ensure package is installed
   pip install grcm-resonant

   # Or install from source
   pip install -e .

**torch.compile not available**:

.. code-block:: bash

   # Requires PyTorch 2.0+
   pip install --upgrade torch

**CUDA out of memory**:

.. code-block:: python

   # Reduce batch size or use CPU
   model = ResonantConsciousnessModule(512, 768, 128)  # Smaller hidden_dim
   result = model(image_emb, audio_emb)  # CPU inference

**Docker build fails**:

.. code-block:: bash

   # Increase Docker memory limit
   # Docker Desktop → Settings → Resources → Memory: 4GB+

   # Clean build
   docker build --no-cache -t grcm-resonant .

Next Steps
----------

- :doc:`quickstart` - Get started with basic usage
- :doc:`configuration` - Configure GRCM via YAML
- :doc:`../tutorials/basic_usage` - Detailed tutorial
