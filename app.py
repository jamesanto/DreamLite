"""
DreamLite Gradio App — unified text-to-image generation and text-guided editing.

Supports both DreamLite-base (28-step, high fidelity) and DreamLite-mobile (4-step, fast).
Includes optional 4-bit quantization and pipeline optimizations for low-VRAM GPUs.
"""

import argparse
import faulthandler
import logging
import os
import time

print("Starting DreamLite... (loading libraries)")

import diffusers.utils.logging as diffusers_logging
import gradio as gr
import torch
import transformers.utils.logging as transformers_logging
from PIL import Image
from tqdm import tqdm

from dreamlite import DreamLiteMobilePipeline, DreamLitePipeline
from dreamlite.pipelines.dreamlite.face_swap import swap_face
from dreamlite.pipelines.dreamlite.optimize import (
    get_optimal_dtype,
    is_turing_gpu,
)
from dreamlite.pipelines.dreamlite.upscale import upscale_tiled

# Print Python traceback on segfault/fatal signal (no-op if signal unavailable)
faulthandler.enable()

# Prevent pandas from using pyarrow string backend — it segfaults on Windows
# when Gradio's queueing.py calls compute_analytics_summary → DataFrame.
# This MUST be set before pandas is imported (by gradio).
os.environ["PANDAS_FUTURE_INFER_STRING"] = "0"
# Also prevent the analytics code path from ever executing (belt-and-suspenders)
os.environ["GRADIO_ANALYTICS_CACHE_FREQUENCY"] = "1000000000"

# ─── CLI Args (parsed early so they're available to configuration) ────────────

_parser = argparse.ArgumentParser(description="DreamLite Gradio App")
_parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
_parser.add_argument("--port", type=int, default=7863, help="Server port")
_parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
_parser.add_argument("--no-compile", action="store_true", help="Disable CUDA graph acceleration (use eager mode)")
_parser.add_argument(
    "--use-inductor", action="store_true", help="Use torch.compile inductor backend instead of CUDA graphs"
)
_parser.add_argument(
    "--vae-tiling", action="store_true", help="Enable VAE tiling (prevents OOM at high resolutions, adds latency)"
)
APP_ARGS = _parser.parse_args() if __name__ == "__main__" else _parser.parse_args([])

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s\033[0m %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dreamlite.app")

diffusers_logging.set_verbosity_warning()
transformers_logging.set_verbosity_warning()

# Suppress noisy inductor/dynamo warnings (e.g. "Not enough SMs to use max_autotune_gemm")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# ─── Configuration ───────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = get_optimal_dtype() if torch.cuda.is_available() else torch.float32

log.info("Device: %s | Dtype: %s", DEVICE, DTYPE)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log.info("GPU: %s (%.1f GB VRAM) | Turing: %s", gpu_name, vram_gb, is_turing_gpu())

MODEL_REGISTRY = {
    "DreamLite-mobile (4 steps, fast)": {
        "path": "models/DreamLite-mobile",
        "cls": DreamLiteMobilePipeline,
        "default_steps": 4,
        "default_guidance": 1.0,
        "supports_cfg": False,
    },
    "DreamLite-base (28 steps, high quality)": {
        "path": "models/DreamLite-base",
        "cls": DreamLitePipeline,
        "default_steps": 28,
        "default_guidance": 3.5,
        "supports_cfg": True,
    },
}

RESOLUTIONS = [
    "1024 × 1024 (1:1)",
    "1152 × 896 (9:7)",
    "896 × 1152 (7:9)",
    "1216 × 832 (3:2)",
    "832 × 1216 (2:3)",
    "1344 × 768 (16:9)",
    "768 × 1344 (9:16)",
]

EDIT_BUCKETS = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
]


def _prepare_edit_image(img: Image.Image) -> tuple[Image.Image, int, int, tuple[int, int, int, int]]:
    """
    Resize and pad an input image to the best matching bucket resolution.

    Strategy:
      1. Pick the bucket whose aspect ratio is closest to the input.
      2. Scale the image so its longest side matches the bucket's longest side.
      3. Pad the shorter side symmetrically (gray fill) to reach exact bucket dims.

    Returns:
        (padded_image, bucket_w, bucket_h, crop_box)
        where crop_box = (left, top, right, bottom) to remove padding from output.
    """
    img_w, img_h = img.size
    target_ar = img_w / img_h
    bucket_w, bucket_h = min(EDIT_BUCKETS, key=lambda b: abs((b[0] / b[1]) - target_ar))

    img_long = max(img_w, img_h)
    bucket_long = max(bucket_w, bucket_h)
    scale = bucket_long / img_long

    new_w = round(img_w * scale)
    new_h = round(img_h * scale)

    # Clamp to bucket dimensions (shouldn't exceed, but safety)
    new_w = min(new_w, bucket_w)
    new_h = min(new_h, bucket_h)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Pad symmetrically to fill bucket
    pad_left = (bucket_w - new_w) // 2
    pad_top = (bucket_h - new_h) // 2

    padded = Image.new("RGB", (bucket_w, bucket_h), (128, 128, 128))
    padded.paste(resized, (pad_left, pad_top))

    crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)

    return padded, bucket_w, bucket_h, crop_box


# ─── State ───────────────────────────────────────────────────────────────────

_pipeline_cache: dict = {}
_active_model: str | None = None
_cancel_requested: bool = False


def _parse_resolution(res_str: str) -> tuple[int, int]:
    parts = res_str.split("(")[0].strip().split("×")
    return int(parts[0].strip()), int(parts[1].strip())


def _load_pipeline(model_name: str):
    global _active_model

    if model_name in _pipeline_cache:
        _active_model = model_name
        pipe = _pipeline_cache[model_name]
        if torch.cuda.is_available():
            if hasattr(pipe, "unet") and next(pipe.unet.parameters()).device.type != "cuda":
                log.info("Moving UNet + VAE back to GPU...")
                pipe.unet.to(DEVICE)
                if hasattr(pipe, "vae"):
                    pipe.vae.to(DEVICE)
        log.info("Using cached pipeline: %s", model_name)
        return pipe

    if _active_model and _active_model != model_name and _active_model in _pipeline_cache:
        log.info("Unloading %s to free memory...", _active_model)
        del _pipeline_cache[_active_model]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = MODEL_REGISTRY[model_name]
    load_kwargs: dict = {"torch_dtype": DTYPE}

    log.info("Loading %s (fp16 text encoder, CPU offload after encode)...", model_name)

    t0 = time.perf_counter()
    pipe = config["cls"].from_pretrained(config["path"], **load_kwargs)
    pipe = pipe.to(DEVICE)

    if hasattr(pipe, "optimize") and DEVICE == "cuda":
        use_cuda_graph = not APP_ARGS.no_compile and not APP_ARGS.use_inductor
        use_inductor = APP_ARGS.use_inductor and not APP_ARGS.no_compile
        pipe.optimize(
            offload_text_encoder=True,
            compile_unet_model=use_inductor,
            use_cuda_graph=use_cuda_graph,
            fuse_qkv=False,
            enable_vae_tiling=APP_ARGS.vae_tiling,
        )

    pipe.set_progress_bar_config(disable=True)

    elapsed = time.perf_counter() - t0
    log.info("Pipeline ready in %.1fs", elapsed)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        log.info("VRAM: %.2f GB allocated, %.2f GB reserved", allocated, reserved)

    _pipeline_cache[model_name] = pipe
    _active_model = model_name
    return pipe


# ─── Inference ───────────────────────────────────────────────────────────────


def _cancel_generation():
    """Called when the user clicks Stop."""
    global _cancel_requested
    _cancel_requested = True
    log.info("Cancel requested by user.")
    return gr.update(interactive=False, value="Stopping...")


def generate(
    model_name: str,
    prompt: str,
    input_image: Image.Image | None,
    resolution: str,
    steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    seed: int,
    upscale_4x: bool,
    restore_face: bool,
    progress=gr.Progress(),
):
    global _cancel_requested
    _cancel_requested = False

    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    task = "edit" if input_image is not None else "generate"
    log.info("─" * 60)
    log.info("Task: %s | Model: %s", task, model_name)
    log.info("Prompt: %s", prompt[:80] + ("..." if len(prompt) > 80 else ""))
    log.info("Steps: %d | Guidance: %.1f | Seed: %d", steps, guidance_scale, seed)

    progress(0, desc="Loading model...")
    yield "Loading model...", None
    pipe = _load_pipeline(model_name)

    if seed < 0:
        seed = torch.randint(0, 2**32, (1,)).item()
        log.info("Random seed: %d", seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    width, height = _parse_resolution(resolution)
    crop_box = None
    original_input = input_image
    if input_image is not None:
        padded_img, width, height, crop_box = _prepare_edit_image(input_image)
        log.info(
            "Edit: input %d×%d → resized+padded to %d×%d (crop_box=%s)",
            *input_image.size,
            width,
            height,
            crop_box,
        )
        input_image = padded_img
    else:
        log.info("Resolution: %d × %d", width, height)

    kwargs = {
        "prompt": prompt,
        "image": input_image,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "generator": generator,
    }

    if input_image is not None:
        kwargs["bucket"] = -1

    config = MODEL_REGISTRY[model_name]
    if config["supports_cfg"]:
        kwargs["guidance_scale"] = guidance_scale
        if input_image is not None:
            kwargs["image_guidance_scale"] = image_guidance_scale

    pbar = tqdm(total=steps, desc="Denoising", unit="step", dynamic_ncols=True)

    def step_callback(step: int, total: int, step_time: float):
        pbar.set_postfix({"s/step": f"{step_time:.2f}"})
        pbar.update(1)
        progress(step / total, desc=f"Step {step}/{total} ({step_time:.1f}s/step)")

    kwargs["callback_on_step_end"] = step_callback
    kwargs["interrupt_flag"] = lambda: _cancel_requested

    progress(0.05, desc="Encoding text...")
    yield "Generating...", None

    t0 = time.perf_counter()
    output = pipe(**kwargs)
    pbar.close()
    elapsed = time.perf_counter() - t0

    if _cancel_requested:
        log.info("Generation cancelled after %.1fs", elapsed)
        yield "Cancelled.", None
        return

    log.info("Generated in %.2fs (%.2f steps/s)", elapsed, steps / elapsed)

    if torch.cuda.is_available():
        try:
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            log.info("Peak VRAM: %.2f GB", peak)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # Offload UNet and VAE to CPU — they are no longer needed for post-processing
    if torch.cuda.is_available():
        log.info("Offloading UNet + VAE to free VRAM for post-processing...")
        if hasattr(pipe, "unet"):
            pipe.unet.to("cpu")
        if hasattr(pipe, "vae"):
            pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    result = output.images[0]
    log.info("Output type: %s", type(result).__name__)

    if not isinstance(result, Image.Image):
        import numpy as np

        if isinstance(result, np.ndarray):
            if result.dtype != np.uint8:
                result = (result.clip(0, 1) * 255).astype(np.uint8)
            result = Image.fromarray(result)
        else:
            raise gr.Error(f"Pipeline returned unexpected output type: {type(result)}")
    if result.mode != "RGB":
        result = result.convert("RGB")

    if crop_box is not None:
        result = result.crop(crop_box)
        log.info("Cropped padding: %s → %s", (width, height), result.size)

    yield f"Denoised in {elapsed:.1f}s", result

    if restore_face and original_input is not None:
        progress(0.85, desc="Restoring face...")
        yield "Restoring face...", result
        log.info("Restoring original face...")
        t_face = time.perf_counter()
        swapped = swap_face(original_input, result)
        if swapped is not None:
            result = swapped
            log.info("Face restored in %.1fs", time.perf_counter() - t_face)
            yield "Face restored", result
        else:
            log.info("No face detected — skipping face restore")

    if upscale_4x:
        progress(0.9, desc="Upscaling 4x...")
        yield "Upscaling 4x...", result
        log.info("Upscaling %s with 4x SPAN...", result.size)
        t_up = time.perf_counter()
        result = upscale_tiled(result, device=torch.device(DEVICE), dtype=DTYPE)
        log.info("Upscaled to %s in %.1fs", result.size, time.perf_counter() - t_up)
        yield f"Done — {result.size[0]}×{result.size[1]}", result

    log.info("Image: %s mode=%s", result.size, result.mode)

    log.info("Done in %.1fs (%.1f steps/s)", elapsed, steps / elapsed)
    progress(1.0, desc=f"Done in {elapsed:.1f}s")
    yield f"Done in {elapsed:.1f}s — {result.size[0]}×{result.size[1]}", result


def on_model_change(model_name: str):
    config = MODEL_REGISTRY[model_name]
    return (
        gr.update(value=config["default_steps"]),
        gr.update(value=config["default_guidance"]),
        gr.update(interactive=config["supports_cfg"]),
        gr.update(interactive=config["supports_cfg"]),
    )


# ─── UI ──────────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="DreamLite",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            "# DreamLite\n"
            "**Lightweight on-device unified model for image generation and editing.**\n\n"
            "Select a model, enter a prompt, and press **Ctrl+Enter** or click Generate.\n"
            "Upload an image to switch to editing mode."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    value=list(MODEL_REGISTRY.keys())[0],
                    label="Model",
                )
                prompt_input = gr.Textbox(
                    label="Prompt (Ctrl+Enter to generate)",
                    placeholder="Describe the image to generate, or the edit to apply...",
                    lines=3,
                    submit_btn=True,
                )
                image_input = gr.Image(
                    type="pil",
                    label="Source Image (optional, for editing)",
                )
                resolution_dropdown = gr.Dropdown(
                    choices=RESOLUTIONS,
                    value=RESOLUTIONS[0],
                    label="Resolution",
                )

                with gr.Accordion("Advanced", open=False):
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=4,
                        step=1,
                        label="Inference Steps",
                    )
                    guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=1.0,
                        step=0.1,
                        label="Guidance Scale",
                    )
                    img_guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="Image Guidance Scale",
                    )
                    seed_input = gr.Number(
                        value=-1,
                        label="Seed (-1 = random)",
                        precision=0,
                    )
                    upscale_checkbox = gr.Checkbox(
                        value=True,
                        label="4x Upscale (SPAN)",
                    )
                    face_restore_checkbox = gr.Checkbox(
                        value=True,
                        label="Restore Face (swap original face back after edit)",
                    )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                stop_btn = gr.Button("Stop", variant="stop", size="lg", visible=True)

            with gr.Column(scale=1):
                progress_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )
                output_image = gr.Image(label="Result", format="png")

        # ─── Event bindings ──────────────────────────────────────────────

        all_inputs = [
            model_dropdown,
            prompt_input,
            image_input,
            resolution_dropdown,
            steps_slider,
            guidance_slider,
            img_guidance_slider,
            seed_input,
            upscale_checkbox,
            face_restore_checkbox,
        ]

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[steps_slider, guidance_slider, guidance_slider, img_guidance_slider],
        )

        # Button click
        gen_event = generate_btn.click(
            fn=generate,
            inputs=all_inputs,
            outputs=[progress_text, output_image],
        )

        # Ctrl+Enter / submit from the prompt textbox
        submit_event = prompt_input.submit(
            fn=generate,
            inputs=all_inputs,
            outputs=[progress_text, output_image],
        )

        # Stop button cancels the running generation
        stop_btn.click(
            fn=_cancel_generation,
            inputs=None,
            outputs=[stop_btn],
            cancels=[gen_event, submit_event],
        )

        gr.Examples(
            examples=[
                [
                    list(MODEL_REGISTRY.keys())[0],
                    "A portrait of a young woman with flowers",
                    None,
                    "1024 × 1024 (1:1)",
                    4,
                    1.0,
                    1.0,
                    -1,
                    True,
                    True,
                ],
                [
                    list(MODEL_REGISTRY.keys())[1],
                    "A close-up of a fire-breathing dragon, cinematic shot",
                    None,
                    "832 × 1216 (2:3)",
                    28,
                    3.5,
                    1.0,
                    -1,
                    True,
                    True,
                ],
            ],
            inputs=all_inputs,
        )

    return app


if __name__ == "__main__":
    if APP_ARGS.no_compile:
        log.info("Acceleration disabled via --no-compile (eager mode)")
    elif APP_ARGS.use_inductor:
        log.info("Using torch.compile inductor backend (via --use-inductor)")
    else:
        log.info("Using CUDA graph acceleration (default)")
    log.info("Starting DreamLite app on %s:%d", APP_ARGS.host, APP_ARGS.port)
    app = build_app()
    app.launch(
        server_name=APP_ARGS.host,
        server_port=APP_ARGS.port,
        share=APP_ARGS.share,
        theme=gr.themes.Soft(),
    )
