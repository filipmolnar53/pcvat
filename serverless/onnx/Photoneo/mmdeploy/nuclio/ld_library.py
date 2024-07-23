_CUDA_PATH = "/home/miro/photoneo/lib/cuda-11.8/lib64"
_CUDNN_PATH = "/home/miro/photoneo/lib/cudnn-linux-x86_64-8.9.2.26_cuda11-archive/lib"
_TENSORRT_PATH = "/home/miro/photoneo/lib/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/lib"
_OPENVINO_PATH = (
    "/home/miro/photoneo/lib/l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64/runtime/lib/intel64"
)

print(":".join([_CUDA_PATH, _CUDNN_PATH, _TENSORRT_PATH, _OPENVINO_PATH]))

_CUDA_PATH = "/home/miro/photoneo/lib_docker/cuda-11.8.0"
_CUDNN_PATH = "/home/miro/photoneo/lib_docker/cudnn-8.6.2"
_TENSORRT_PATH = "/home/miro/photoneo/lib_docker/tensorrt-8.6"
_OPENVINO_PATH = "/home/miro/photoneo/lib_docker/openvino-2024.1"

print(":".join([_CUDA_PATH, _CUDNN_PATH, _TENSORRT_PATH, _OPENVINO_PATH]))
