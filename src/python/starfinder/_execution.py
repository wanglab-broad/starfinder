"""Cross-stage execution device and the execution record (see the spot-finding contract).

The device belongs to execution, not to a method, so every stage that runs a
method checks and records it the same way. Starfinder does not change thread
settings, nor torch's TF32 settings on a GPU; the record shows what was in effect.
"""
import os
import sys

DEVICES = ("cpu", "cuda")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "NUMBA_NUM_THREADS")


def check_device(device) -> None:
    """Raise ValueError unless device is "cpu" or "cuda"; which methods accept "cuda" is the method's own check."""
    # §2.7; "cuda" for the learned detectors, W-796
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}")


def execution_record(device: str, *, framework: bool = False) -> dict:
    """The execution entry of one method invocation.

    framework is True for methods that run on torch: the entry then records
    torch's version, its CUDA build (None for CPU builds) and whether MKL and
    MKL-DNN are available; otherwise framework is None. threads records
    torch's intra- and inter-op thread counts when torch is already imported
    (None otherwise; this never imports torch) and the thread variables of
    the environment (None when unset). On "cuda" a torch entry also records
    the GPU's name and torch's TF32 settings (matmul and cuDNN), which can
    change a result against the CPU.
    """
    check_device(device)
    torch = sys.modules.get("torch")
    entry = None
    if framework and torch is not None:
        entry = {"name": "torch", "version": str(torch.__version__), "cuda": torch.version.cuda,
                 "mkl": bool(torch.backends.mkl.is_available()), "mkldnn": bool(torch.backends.mkldnn.is_available())}
        if device == "cuda":
            entry.update(gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
                         tf32={"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                               "cudnn": bool(torch.backends.cudnn.allow_tf32)})
    threads = {"torch_num_threads": torch.get_num_threads() if torch is not None else None,
               "torch_num_interop_threads": torch.get_num_interop_threads() if torch is not None else None}
    threads.update({name: os.environ.get(name) for name in THREAD_VARIABLES})
    return {"device": device, "framework": entry, "threads": threads}
