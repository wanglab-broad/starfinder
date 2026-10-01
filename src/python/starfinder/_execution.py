"""Cross-stage execution device and the execution record (see the spot-finding contract).

The device belongs to execution, not to a method, so every stage that runs a
method checks and records it the same way. Starfinder does not change thread
settings; the record shows what was in effect.
"""
import os
import sys

DEVICES = ("cpu",)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "NUMBA_NUM_THREADS")


def check_device(device) -> None:
    """Raise ValueError unless device is "cpu", the only device of §2.7."""
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError("device must be 'cpu'; §2.7 runs on CPU only")


def execution_record(device: str, *, framework: bool = False) -> dict:
    """The execution entry of one method invocation.

    framework is True for methods that run on torch: the entry then records
    torch's version, its CUDA build (None for CPU builds) and whether MKL and
    MKL-DNN are available; otherwise framework is None. threads records
    torch's intra- and inter-op thread counts when torch is already imported
    (None otherwise; this never imports torch) and the thread variables of
    the environment (None when unset).
    """
    check_device(device)
    torch = sys.modules.get("torch")
    entry = None
    if framework and torch is not None:
        entry = {"name": "torch", "version": str(torch.__version__), "cuda": torch.version.cuda,
                 "mkl": bool(torch.backends.mkl.is_available()), "mkldnn": bool(torch.backends.mkldnn.is_available())}
    threads = {"torch_num_threads": torch.get_num_threads() if torch is not None else None,
               "torch_num_interop_threads": torch.get_num_interop_threads() if torch is not None else None}
    threads.update({name: os.environ.get(name) for name in THREAD_VARIABLES})
    return {"device": device, "framework": entry, "threads": threads}
