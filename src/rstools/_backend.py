import numpy as np

try:
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() > 0
    HAS_GPU = True
except Exception:
    cp = None
    HAS_GPU = False


def get_backend(*args):
    """
    Returns cupy if any arg is on GPU (or if no args are passed and GPU is available).
    Otherwise returns numpy.
    """
    if HAS_GPU:
        if not args:
            return cp
        return cp.get_array_module(*args)
    return np


def to_cpu(x):
    """Always returns a numpy array on CPU."""
    return x.get() if (HAS_GPU and isinstance(x, cp.ndarray)) else np.asarray(x)


def to_gpu(x):
    """Always returns a cupy array on GPU (falls back to numpy if no GPU)."""
    return cp.asarray(x) if HAS_GPU else np.asarray(x)


__all__ = ["HAS_GPU", "get_backend", "to_cpu", "to_gpu"]
