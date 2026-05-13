"""
Face swap module: restore the original face from the input image into the output.

After DreamLite edits an image, the model may alter facial features. This module
detects the face in the source (input) image and swaps it back onto the detected
face in the output, preserving the original identity.

Uses insightface for detection/analysis and the inswapper_128 model for swapping.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_HF_REPO = "mikestealth/inswapper"
_HF_FILENAME = "inswapper_128.onnx"
_FACE_ANALYSER = None
_FACE_SWAPPER = None


def offload_face_swap():
    """Release face swap models to free GPU memory (ONNX Runtime sessions)."""
    global _FACE_ANALYSER, _FACE_SWAPPER
    _FACE_ANALYSER = None
    _FACE_SWAPPER = None


def _download_swapper_model() -> Path:
    """Download inswapper_128.onnx from HuggingFace (cached after first download)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
    logger.info("inswapper_128 model at: %s", path)
    return Path(path)


def _get_face_analyser():
    """Load and cache the insightface face analyser (buffalo_l model)."""
    global _FACE_ANALYSER
    if _FACE_ANALYSER is not None:
        return _FACE_ANALYSER

    from insightface.app import FaceAnalysis

    _FACE_ANALYSER = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("insightface FaceAnalysis loaded (buffalo_l)")
    return _FACE_ANALYSER


def _get_face_swapper():
    """Load and cache the inswapper model."""
    global _FACE_SWAPPER
    if _FACE_SWAPPER is not None:
        return _FACE_SWAPPER

    import insightface

    model_path = _download_swapper_model()
    _FACE_SWAPPER = insightface.model_zoo.get_model(str(model_path))
    logger.info("inswapper_128 face swapper loaded")
    return _FACE_SWAPPER


def _get_largest_face(faces):
    """Return the face with the largest bounding box area."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def swap_face(
    source_image: Image.Image,
    target_image: Image.Image,
) -> Optional[Image.Image]:
    """
    Swap the face from source_image onto target_image.

    Detects the largest face in each image. If either image has no detectable
    face, returns None (no swap performed).

    Args:
        source_image: The original input image (has the face we want to keep).
        target_image: The generated/edited output (has the face we want to replace).

    Returns:
        The target image with the source face swapped in, or None if no faces detected.
    """
    analyser = _get_face_analyser()
    swapper = _get_face_swapper()

    source_arr = np.array(source_image.convert("RGB"))
    target_arr = np.array(target_image.convert("RGB"))

    source_faces = analyser.get(source_arr)
    source_face = _get_largest_face(source_faces)
    if source_face is None:
        logger.warning("No face detected in source image — skipping face swap")
        return None

    target_faces = analyser.get(target_arr)
    target_face = _get_largest_face(target_faces)
    if target_face is None:
        logger.warning("No face detected in output image — skipping face swap")
        return None

    logger.info("Swapping face (source confidence=%.2f, target confidence=%.2f)", source_face.det_score, target_face.det_score)

    result = swapper.get(target_arr, target_face, source_face, paste_back=True)
    return Image.fromarray(result)
