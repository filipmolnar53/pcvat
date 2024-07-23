Install Python 3.11 and requirements for GPU version.

Then install one of the following:

```
# CPU
onnxruntime
# CUDA
onnxruntime-gpu
# OpenVino
onnxruntime-openvino
```

For TensorRT you must use a custom compiled wheel.

For CPU or OpenVINO, run `run_onnxruntime.py`.
For CUDA or TensorRT, install/copy CUDA libraries (cuda, cudnn, tensorrt).
Run `run_onnxruntime.py` with `LD_LIBRARY_PATH` set to the lib directories (`ld_library.py` can help).

MMDetectrion TensorRT model specific - use `rewrite_onnx.py`.
