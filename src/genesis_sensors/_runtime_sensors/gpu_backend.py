"""GPU backend selection for sensor processing acceleration.

Auto-detects Metal MPS on macOS, CUDA on Linux, CPU fallback.
Uses PyTorch as the tensor backend (already a project dependency).
"""

from __future__ import annotations

import logging
import platform
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GPU_DEVICE: Any = None
_GPU_ENABLED: bool = False
_TORCH: Any = None


def _detect_backend() -> str:
    """Auto-detect the best available GPU backend."""
    global _TORCH
    system = platform.system()
    try:
        import torch as t
        _TORCH = t
        if system == "Darwin" and t.backends.mps.is_available():
            return "mps"
        elif t.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def init_gpu(backend: str = "auto", enable: bool = True) -> str:
    """Initialize the GPU backend.

    Args:
        backend: "auto", "mps", "cuda", "cpu"
        enable: If False, force CPU regardless.

    Returns:
        The selected backend name.
    """
    global _GPU_DEVICE, _GPU_ENABLED

    if not enable:
        _GPU_DEVICE = None
        _GPU_ENABLED = False
        logger.debug("GPU disabled — using CPU")
        return "cpu"

    if backend == "auto":
        backend = _detect_backend()

    if backend in ("mps", "cuda"):
        try:
            _GPU_DEVICE = _TORCH.device(backend)
            # Warm up by creating a small tensor
            _TORCH.zeros(1, device=_GPU_DEVICE)
            _GPU_ENABLED = True
            logger.info("GPU backend: %s", backend)
            return backend
        except Exception:
            logger.warning("GPU backend %s failed — using CPU", backend, exc_info=True)

    _GPU_DEVICE = None
    _GPU_ENABLED = False
    logger.debug("Using CPU backend")
    return "cpu"


def get_device() -> Any:
    """Get the current GPU device, or None for CPU."""
    return _GPU_DEVICE


def is_gpu_enabled() -> bool:
    """Whether GPU acceleration is active."""
    return _GPU_ENABLED


def to_tensor(arr: np.ndarray, dtype: Any = None) -> Any:
    """Convert numpy array to tensor on the current GPU device (or CPU)."""
    global _TORCH
    if _TORCH is None:
        import torch as t
        _TORCH = t
    if dtype is not None:
        arr = arr.astype(dtype)
    tensor = _TORCH.from_numpy(arr)
    if _GPU_ENABLED and _GPU_DEVICE is not None:
        return tensor.to(_GPU_DEVICE)
    return tensor


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor to numpy array, moving from GPU if needed."""
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def tensor_zeros(shape: tuple[int, ...], dtype: Any = None) -> Any:
    """Create a zero tensor on GPU or CPU."""
    if dtype is None:
        dtype = _TORCH.float32 if _TORCH is not None else np.float32
    if _GPU_ENABLED and _GPU_DEVICE is not None:
        return _TORCH.zeros(shape, dtype=dtype, device=_GPU_DEVICE)
    if _TORCH is not None:
        return _TORCH.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=np.dtype(dtype) if hasattr(dtype, 'name') else dtype)
