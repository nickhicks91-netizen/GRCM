Optimization Module
===================

.. automodule:: grcm.optimization
   :members:
   :undoc-members:
   :show-inheritance:

OptimizedGRCM
-------------

.. autoclass:: grcm.optimization.OptimizedGRCM
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Optimized GRCM wrapper with torch.compile, quantization, and mixed precision.

   **Optimization Modes**:

   +-------------------+---------------+------------------+-------------------+
   | Mode              | Speedup       | Quality Impact   | Use Case          |
   +===================+===============+==================+===================+
   | torch.compile     | 2.5x          | None             | Production CPU    |
   +-------------------+---------------+------------------+-------------------+
   | INT8 Quantize     | 1.8x          | Minimal (<1%)    | Edge devices      |
   +-------------------+---------------+------------------+-------------------+
   | FP16 (GPU)        | 1.5-2x        | None             | Production GPU    |
   +-------------------+---------------+------------------+-------------------+
   | TensorRT FP16     | 14x           | Minimal (<2%)    | NVIDIA GPUs       |
   +-------------------+---------------+------------------+-------------------+

   **Example**:

   .. code-block:: python

      from grcm import ResonantConsciousnessModule
      from grcm.optimization import OptimizedGRCM

      model = ResonantConsciousnessModule(512, 768, 256)

      # torch.compile optimization (CPU/GPU)
      opt_model = OptimizedGRCM(
          model,
          mode='reduce-overhead',  # or 'max-autotune'
          quantize=False,
          use_fp16=False
      )

      # Forward pass (automatically optimized)
      result = opt_model(image_emb, audio_emb)

   **INT8 Quantization** (CPU only):

   .. code-block:: python

      opt_model = OptimizedGRCM(
          model,
          mode='reduce-overhead',
          quantize=True,  # Enable INT8
          use_fp16=False
      )

   **FP16 Mixed Precision** (GPU only):

   .. code-block:: python

      opt_model = OptimizedGRCM(
          model,
          mode='max-autotune',
          quantize=False,
          use_fp16=True  # Enable FP16
      )

   **Performance Tips**:

   - Use ``mode='max-autotune'`` for best throughput (longer compile time)
   - Use ``mode='reduce-overhead'`` for balanced compile time vs speed
   - Enable ``quantize=True`` for CPU-only deployments
   - Enable ``use_fp16=True`` for GPU deployments with Tensor Cores

   **Methods**:

   .. automethod:: forward
   .. automethod:: benchmark

ONNX Export
-----------

Export GRCM to ONNX format for cross-platform deployment:

.. code-block:: python

   from grcm.optimization import export_to_onnx

   export_to_onnx(
       model,
       output_path='grcm_model.onnx',
       opset_version=18,
       dynamic_batch=True
   )

   # Inference with ONNX Runtime
   import onnxruntime as ort

   session = ort.InferenceSession('grcm_model.onnx')
   outputs = session.run(None, {
       'image_emb': image_emb.numpy(),
       'audio_emb': audio_emb.numpy()
   })

TensorRT Conversion
-------------------

For maximum GPU performance, convert to TensorRT:

.. code-block:: bash

   # 1. Export to ONNX
   python scripts/export_onnx.py --output grcm.onnx

   # 2. Convert to TensorRT FP16
   trtexec --onnx=grcm.onnx \
           --saveEngine=grcm_fp16.trt \
           --fp16 \
           --minShapes=image_emb:1x512,audio_emb:1x768 \
           --optShapes=image_emb:8x512,audio_emb:8x768 \
           --maxShapes=image_emb:32x512,audio_emb:32x768

See :doc:`/docs/TENSORRT_GUIDE` for detailed TensorRT conversion instructions.
