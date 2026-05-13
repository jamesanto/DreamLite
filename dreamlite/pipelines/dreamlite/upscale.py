"""
4x SPAN upscaler with tiled inference for low-VRAM GPUs.

Uses 4xNomosUni_span_multijpg — a fast SPAN-architecture model optimized
for realistic photos. ~5-8x faster than RRDB (UltraSharp) with comparable
quality. Downloads from HuggingFace on first use.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_HF_REPO = "Phips/4xNomosUni_span_multijpg"
_HF_FILENAME = "4xNomosUni_span_multijpg.pth"
_MODEL_CACHE: dict = {}


def _download_model() -> Path:
    """Download SPAN upscaler weights from HuggingFace (cached after first download)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
    logger.info("4xNomosUni_span_multijpg model at: %s", path)
    return Path(path)


def _load_model(device: torch.device, dtype: torch.dtype):
    """Load the upscaler model (cached in memory after first load)."""
    cache_key = (str(device), str(dtype))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from spandrel import ImageModelDescriptor, ModelLoader

    model_path = _download_model()
    model = ModelLoader().load_from_file(str(model_path))
    assert isinstance(model, ImageModelDescriptor)

    use_half = dtype == torch.float16 and model.supports_half
    model = model.to(device)
    if use_half:
        model.model.half()
    model.eval()

    _MODEL_CACHE[cache_key] = (model, use_half)
    logger.info("SPAN upscaler loaded (device=%s, half=%s, scale=%dx)", device, use_half, model.scale)
    return model, use_half


def _img_to_tensor(img: Image.Image, device: torch.device, half: bool) -> torch.Tensor:
    """Convert PIL Image to BCHW float tensor in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if half:
        tensor = tensor.half()
    return tensor


def _tensor_to_img(tensor: torch.Tensor) -> Image.Image:
    """Convert BCHW tensor back to PIL Image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


@torch.inference_mode()
def upscale_tiled(
    img: Image.Image,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
    tile_size: int = 512,
    tile_pad: int = 16,
) -> Image.Image:
    """
    Upscale an image 4x using tiled inference.

    Args:
        img: Input PIL Image (RGB).
        device: CUDA device (defaults to cuda:0 if available).
        dtype: Inference dtype (fp16 recommended for speed/VRAM).
        tile_size: Size of each processing tile (default 512).
        tile_pad: Overlap padding between tiles to avoid seam artifacts.

    Returns:
        Upscaled PIL Image (4x resolution).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, use_half = _load_model(device, dtype)
    scale = model.scale

    img = img.convert("RGB")
    w, h = img.size
    out_h, out_w = h * scale, w * scale

    input_tensor = _img_to_tensor(img, device, use_half)
    _, _, img_h, img_w = input_tensor.shape

    output = torch.zeros((1, 3, out_h, out_w), dtype=input_tensor.dtype, device=device)

    tiles_x = max(1, (img_w + tile_size - 1) // tile_size)
    tiles_y = max(1, (img_h + tile_size - 1) // tile_size)

    logger.info("Upscaling %dx%d → %dx%d (%dx%d tiles)", w, h, out_w, out_h, tiles_x, tiles_y)

    for y_idx in range(tiles_y):
        for x_idx in range(tiles_x):
            x_start = x_idx * tile_size
            y_start = y_idx * tile_size
            x_end = min(x_start + tile_size, img_w)
            y_end = min(y_start + tile_size, img_h)

            x_start_pad = max(x_start - tile_pad, 0)
            y_start_pad = max(y_start - tile_pad, 0)
            x_end_pad = min(x_end + tile_pad, img_w)
            y_end_pad = min(y_end + tile_pad, img_h)

            tile_input = input_tensor[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad]
            tile_output = model(tile_input)

            out_x_start = (x_start - x_start_pad) * scale
            out_y_start = (y_start - y_start_pad) * scale
            out_x_end = out_x_start + (x_end - x_start) * scale
            out_y_end = out_y_start + (y_end - y_start) * scale

            output[
                :, :,
                y_start * scale : y_end * scale,
                x_start * scale : x_end * scale,
            ] = tile_output[:, :, out_y_start:out_y_end, out_x_start:out_x_end]

    result = _tensor_to_img(output)

    del input_tensor, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
