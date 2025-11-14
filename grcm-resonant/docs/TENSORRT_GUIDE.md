# TensorRT Optimization Guide for GRCM

## Overview
TensorRT provides **5-10x speedup** over PyTorch on NVIDIA GPUs, making GRCM suitable for real-time BCI, robotics, and edge AI applications.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (GTX 1060+, RTX series, Jetson devices)
- Minimum 4GB GPU memory (8GB+ recommended)

### Software
```bash
# CUDA Toolkit 11.8+ or 12.x
# TensorRT 8.6+ (included in JetPack for Jetson)

# Install Python packages
pip install tensorrt
pip install pycuda
pip install onnx onnx-simplifier
```

For Jetson devices:
```bash
# Use JetPack SDK (includes TensorRT)
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
```

## Conversion Pipeline

### Step 1: Export to ONNX

```bash
# Export with dynamic batch axes
python scripts/export_onnx.py \
    --output grcm_model.onnx \
    --simplify \
    --test

# Output: grcm_model.onnx (~2-5 MB)
```

### Step 2: Convert ONNX to TensorRT Engine

```python
#!/usr/bin/env python3
"""convert_tensorrt.py - ONNX to TensorRT conversion"""
import tensorrt as trt
import numpy as np

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 32
):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        fp16: Enable FP16 precision (2x speedup, minimal accuracy loss)
        int8: Enable INT8 precision (requires calibration)
        max_batch_size: Maximum batch size for optimization
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Loading ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    print(f"✓ ONNX model loaded")

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Precision
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 precision enabled")
    elif fp16:
        print("⚠ FP16 not supported on this device, using FP32")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("✓ INT8 precision enabled (requires calibration)")
        # Note: INT8 requires calibration dataset (see below)

    # Optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()

    # Set shape ranges: [min, opt, max]
    profile.set_shape('image_emb', (1, 512), (4, 512), (max_batch_size, 512))
    profile.set_shape('audio_emb', (1, 768), (4, 768), (max_batch_size, 768))
    profile.set_shape('action', (1, 4), (4, 4), (max_batch_size, 4))

    config.add_optimization_profile(profile)

    # Build engine
    print("Building TensorRT engine (this may take 2-5 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ TensorRT engine saved to {engine_path}")
    print(f"  Precision: {'FP16' if fp16 else 'FP32'}")
    print(f"  Max Batch: {max_batch_size}")

# Usage
build_engine('grcm_model.onnx', 'grcm.trt', fp16=True)
```

### Step 3: Inference with TensorRT

```python
#!/usr/bin/env python3
"""tensorrt_inference.py - Run inference with TensorRT engine"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Dict, List

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTGRCM:
    """TensorRT inference wrapper for GRCM."""

    def __init__(self, engine_path: str):
        # Load engine
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': tensor_name, 'dtype': dtype})
            else:
                self.outputs.append({'name': tensor_name, 'dtype': dtype})

        print(f"✓ TensorRT engine loaded")
        print(f"  Inputs: {[inp['name'] for inp in self.inputs]}")
        print(f"  Outputs: {[out['name'] for out in self.outputs]}")

    def __call__(
        self,
        image_emb: np.ndarray,
        audio_emb: np.ndarray,
        action: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Run inference.

        Args:
            image_emb: (batch, 512) numpy array
            audio_emb: (batch, 768) numpy array
            action: (batch, 4) numpy array

        Returns:
            Dict with output tensors
        """
        batch_size = image_emb.shape[0]

        # Set input shapes
        self.context.set_input_shape('image_emb', image_emb.shape)
        self.context.set_input_shape('audio_emb', audio_emb.shape)
        self.context.set_input_shape('action', action.shape)

        # Allocate device memory
        bindings = []
        host_inputs = [image_emb, audio_emb, action]
        device_inputs = []
        device_outputs = []

        for inp, host_input in zip(self.inputs, host_inputs):
            device_input = cuda.mem_alloc(host_input.nbytes)
            cuda.memcpy_htod_async(device_input, host_input, self.stream)
            device_inputs.append(device_input)
            bindings.append(int(device_input))

        for out in self.outputs:
            shape = self.context.get_tensor_shape(out['name'])
            size = trt.volume(shape)
            host_output = np.empty(size, dtype=out['dtype'])
            device_output = cuda.mem_alloc(host_output.nbytes)
            device_outputs.append({'device': device_output, 'host': host_output})
            bindings.append(int(device_output))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs
        results = {}
        for out, dev_out in zip(self.outputs, device_outputs):
            cuda.memcpy_dtoh_async(dev_out['host'], dev_out['device'], self.stream)
            results[out['name']] = dev_out['host']

        self.stream.synchronize()

        return results

# Usage example
if __name__ == "__main__":
    model = TensorRTGRCM('grcm.trt')

    # Test inference
    image_emb = np.random.randn(4, 512).astype(np.float32)
    audio_emb = np.random.randn(4, 768).astype(np.float32)
    action = np.random.randn(4, 4).astype(np.float32)

    results = model(image_emb, audio_emb, action)

    print("\nOutput shapes:")
    for key, val in results.items():
        print(f"  {key}: {val.shape}")
```

## INT8 Calibration (Advanced)

For INT8 quantization, provide a calibration dataset:

```python
import tensorrt as trt
import numpy as np

class GRCMCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator for INT8 quantization."""

    def __init__(self, calibration_data: List[Dict[str, np.ndarray]], cache_file: str = "calibration.cache"):
        super().__init__()
        self.data = calibration_data
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0

        # Allocate device memory for calibration
        self.device_inputs = {}
        for key in ['image_emb', 'audio_emb', 'action']:
            shape = calibration_data[0][key].shape
            self.device_inputs[key] = cuda.mem_alloc(calibration_data[0][key].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None

        batch = self.data[self.current_index]

        # Copy to device
        for key in ['image_emb', 'audio_emb', 'action']:
            cuda.memcpy_htod(self.device_inputs[key], batch[key])

        self.current_index += 1

        return [int(self.device_inputs[key]) for key in ['image_emb', 'audio_emb', 'action']]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# Generate calibration dataset (100-500 samples recommended)
calibration_data = []
for _ in range(100):
    calibration_data.append({
        'image_emb': np.random.randn(1, 512).astype(np.float32),
        'audio_emb': np.random.randn(1, 768).astype(np.float32),
        'action': np.random.randn(1, 4).astype(np.float32)
    })

calibrator = GRCMCalibrator(calibration_data)

# Use in build_engine with int8=True and calibrator
```

## Performance Expectations

### Desktop GPUs

| GPU | PyTorch FP32 | TRT FP16 | TRT INT8 | Speedup |
|-----|--------------|----------|----------|---------|
| RTX 4090 | 20-25 ms | 3-4 ms | 2-3 ms | 8-10x |
| RTX 3090 | 30-40 ms | 5-8 ms | 4-6 ms | 6-8x |
| RTX 3060 | 50-60 ms | 10-12 ms | 8-10 ms | 5-6x |
| GTX 1080 Ti | 60-80 ms | 15-20 ms | N/A | 4-5x |

### Jetson Edge Devices

| Device | PyTorch FP32 | TRT FP16 | TRT INT8 | Power |
|--------|--------------|----------|----------|-------|
| Jetson AGX Orin | 40-50 ms | 8-12 ms | 6-8 ms | 15-30W |
| Jetson AGX Xavier | 80-100 ms | 15-20 ms | 12-15 ms | 10-30W |
| Jetson Orin Nano | 120-150 ms | 25-30 ms | 20-25 ms | 7-15W |
| Jetson Xavier NX | 150-180 ms | 35-40 ms | 30-35 ms | 10-20W |

## Deployment Tips

### 1. Production Serving
```python
# FastAPI server with TensorRT
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = TensorRTGRCM('grcm.trt')

class GRCMInput(BaseModel):
    image_emb: List[List[float]]
    audio_emb: List[List[float]]
    action: List[List[float]]

@app.post("/predict")
def predict(input_data: GRCMInput):
    results = model(
        np.array(input_data.image_emb, dtype=np.float32),
        np.array(input_data.audio_emb, dtype=np.float32),
        np.array(input_data.action, dtype=np.float32)
    )
    return {k: v.tolist() for k, v in results.items()}
```

### 2. BCI/Real-time Systems
- Target: <20ms latency for 50Hz BCI sampling
- Use FP16 on Jetson Orin (12ms @ batch=1)
- Dedicated inference thread with priority scheduling
- Pre-allocate all GPU memory

### 3. Robotics
- TensorRT on NVIDIA Jetson for vision-audio fusion
- Run at 30 Hz control loop (33ms budget)
- FP16 sufficient for proprioceptive feedback

### 4. Edge AI
- Engine size: ~50-100 MB (FP16), ~25-50 MB (INT8)
- RAM: 512 MB minimum (model + activations)
- Storage: Use NVME for faster engine loading

## Troubleshooting

### Graph Conversion Issues
```bash
# Check ONNX compatibility
python -c "import onnx; onnx.checker.check_model(onnx.load('grcm_model.onnx'))"

# Simplify ONNX graph
python scripts/export_onnx.py --output grcm_model.onnx --simplify
```

### Accuracy Validation
```python
# Compare PyTorch vs TensorRT outputs
import torch
from grcm import ResonantConsciousnessModule

# PyTorch
pt_model = ResonantConsciousnessModule(15, 8, 32).eval()
pt_result = pt_model(image_emb_torch, audio_emb_torch, action_torch)

# TensorRT
trt_model = TensorRTGRCM('grcm.trt')
trt_result = trt_model(image_emb_np, audio_emb_np, action_np)

# Compare (FP16 tolerance ±1e-3)
diff = np.abs(pt_result['output'].cpu().numpy() - trt_result['output'])
print(f"Max diff: {diff.max()}")  # Should be <0.001 for FP16
```

### Memory Issues
- Reduce max_batch_size in optimization profile
- Use INT8 to halve memory footprint
- Enable `config.set_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, True)`

## Next Steps

1. **Export ONNX**: `python scripts/export_onnx.py --simplify`
2. **Build Engine**: Run `convert_tensorrt.py` script above
3. **Benchmark**: Compare PyTorch vs TensorRT latency
4. **Deploy**: Integrate TensorRTGRCM into your application

For BentoML/Kubernetes deployment, see Phase 4 documentation.
