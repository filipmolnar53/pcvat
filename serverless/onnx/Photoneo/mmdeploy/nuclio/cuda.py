import os

import pynvml


def print_cuda_memory_usage():
    pynvml.nvmlInit()

    if pynvml.nvmlDeviceGetCount() > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pid = os.getpid()

        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            if process.pid == pid:
                print(f"Using {process.usedGpuMemory / 1024**2:.0f} MB of GPU memory.")
                break

    pynvml.nvmlShutdown()
