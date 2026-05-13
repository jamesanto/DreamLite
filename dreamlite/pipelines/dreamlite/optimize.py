"""
Optimization utilities for DreamLite base pipeline on low-VRAM GPUs.

Provides composable optimization methods targeting RTX 2060 / 6GB VRAM / Windows:
- 4-bit text encoder quantization (bitsandbytes)
- Sequential CPU offloading between pipeline stages
- torch.compile for UNet acceleration
- QKV fusion for attention kernel efficiency
- VAE tiling for OOM prevention during decode
"""

import platform
import sys
from typing import Optional

import torch
from diffusers.utils import logging

logger = logging.get_logger(__name__)

_BNB_AVAILABLE = False
try:
    import bitsandbytes  # noqa: F401
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    pass


def is_turing_gpu() -> bool:
    """Detect NVIDIA Turing architecture (SM 7.x) which lacks native bf16."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 7


def get_optimal_dtype() -> torch.dtype:
    """Return fp16 for Turing GPUs (no native bf16), bf16 for Ampere+."""
    if is_turing_gpu():
        return torch.float16
    return torch.bfloat16


def get_8bit_quantization_config() -> "BitsAndBytesConfig":
    """
    Build a BitsAndBytesConfig for 8-bit quantization of the text encoder.

    8-bit is the sweet spot for 6GB VRAM: nearly lossless quality,
    faster dequant than 4-bit, and still fits comfortably.
    """
    if not _BNB_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for quantization. "
            "Install it with: pip install bitsandbytes"
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )


def get_4bit_quantization_config(compute_dtype: Optional[torch.dtype] = None) -> "BitsAndBytesConfig":
    """
    Build a BitsAndBytesConfig for 4-bit NF4 quantization of the text encoder.

    Raises ImportError if bitsandbytes is not installed.
    """
    if not _BNB_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 4-bit quantization. "
            "Install it with: pip install bitsandbytes"
        )
    if compute_dtype is None:
        compute_dtype = get_optimal_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _triton_available() -> bool:
    """Check if a working Triton installation is present (required for inductor backend)."""
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
        return True
    except ImportError:
        return False


def compile_unet(unet: torch.nn.Module) -> torch.nn.Module:
    """
    Apply torch.compile to the UNet for kernel fusion and CUDA graph capture.

    Uses 'reduce-overhead' mode on Linux, 'default' mode on Windows.
    Requires Triton (triton-windows on Windows) for the inductor backend.
    Falls back to eager mode if Triton is missing or compilation fails.
    """
    if sys.version_info < (3, 10):
        logger.warning("torch.compile requires Python 3.10+; skipping compilation.")
        return unet

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available in this PyTorch version; skipping.")
        return unet

    if not _triton_available():
        logger.warning(
            "Triton not found — torch.compile requires it for the inductor backend. "
            "On Windows, install with: uv add 'triton-windows>=3.2,<3.3' "
            "(must match your PyTorch version). Skipping compilation."
        )
        return unet

    try:
        torch._dynamo.config.suppress_errors = True

        backend = "inductor"
        mode = "reduce-overhead"

        if platform.system() == "Windows":
            mode = "default"

        compiled = torch.compile(unet, mode=mode, backend=backend, fullgraph=False)
        logger.info(f"UNet compiled with backend={backend}, mode={mode}")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile failed ({e}); falling back to eager mode.")
        return unet


def enable_fast_attention() -> None:
    """Enable optimal CUDA attention backends for Turing+ GPUs."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    logger.info("Enabled flash/mem-efficient SDPA and cudnn benchmark.")


def offload_to_cpu(module: torch.nn.Module) -> None:
    """Move a module to CPU and release GPU memory."""
    module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_to_device(module: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    """Move a module back to the target device."""
    module.to(device=device, dtype=dtype)
