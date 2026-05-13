"""
DreamLite Gradio App — unified text-to-image generation and text-guided editing.

Supports both DreamLite-base (28-step, high fidelity) and DreamLite-mobile (4-step, fast).
Includes optional 4-bit quantization and pipeline optimizations for low-VRAM GPUs.
"""

import logging
import time

import gradio as gr
import torch
from PIL import Image

from dreamlite import DreamLiteMobilePipeline, DreamLitePipeline
from dreamlite.pipelines.dreamlite.optimize import (
    _BNB_AVAILABLE,
    get_8bit_quantization_config,
    get_4bit_quantization_config,
    get_optimal_dtype,
    is_turing_gpu,
)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s\033[0m %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dreamlite.app")

# ─── Configuration ───────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = get_optimal_dtype() if torch.cuda.is_available() else torch.float32

log.info("Device: %s | Dtype: %s", DEVICE, DTYPE)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log.info("GPU: %s (%.1f GB VRAM) | Turing: %s", gpu_name, vram_gb, is_turing_gpu())
    log.info("bitsandbytes available: %s", _BNB_AVAILABLE)

MODEL_REGISTRY = {
    "DreamLite-base (20 steps, high quality)": {
        "path": "models/DreamLite-base",
        "cls": DreamLitePipeline,
        "default_steps": 20,
        "default_guidance": 3.5,
        "supports_cfg": True,
    },
    "DreamLite-mobile (4 steps, fast)": {
        "path": "models/DreamLite-mobile",
        "cls": DreamLiteMobilePipeline,
        "default_steps": 4,
        "default_guidance": 1.0,
        "supports_cfg": False,
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

# ─── State ───────────────────────────────────────────────────────────────────

_pipeline_cache: dict = {}
_active_model: str | None = None


def _parse_resolution(res_str: str) -> tuple[int, int]:
    parts = res_str.split("(")[0].strip().split("×")
    return int(parts[0].strip()), int(parts[1].strip())


def _load_pipeline(model_name: str, use_4bit: bool = True):
    global _active_model

    if model_name in _pipeline_cache:
        _active_model = model_name
        log.info("Using cached pipeline: %s", model_name)
        return _pipeline_cache[model_name]

    if _active_model and _active_model != model_name and _active_model in _pipeline_cache:
        log.info("Unloading %s to free memory...", _active_model)
        del _pipeline_cache[_active_model]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = MODEL_REGISTRY[model_name]
    load_kwargs: dict = {"torch_dtype": DTYPE}

    quantized = use_4bit and _BNB_AVAILABLE and DEVICE == "cuda"
    if quantized:
        from transformers import Qwen3VLForConditionalGeneration

        log.info("Loading %s with 8-bit text encoder...", model_name)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            config["path"],
            subfolder="text_encoder",
            quantization_config=get_8bit_quantization_config(),
            device_map="auto",
        )
        load_kwargs["text_encoder"] = text_encoder
    else:
        log.info("Loading %s (full precision)...", model_name)

    t0 = time.perf_counter()
    pipe = config["cls"].from_pretrained(config["path"], **load_kwargs)

    if quantized:
        pipe.unet.to(DEVICE, dtype=DTYPE)
        pipe.vae.to(DEVICE, dtype=DTYPE)
    else:
        pipe = pipe.to(DEVICE)

    if hasattr(pipe, "optimize") and DEVICE == "cuda":
        pipe.optimize(
            offload_text_encoder=True,
            compile_unet_model=True,
            fuse_qkv=False,
            enable_vae_tiling=True,
        )

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


def generate(
    model_name: str,
    prompt: str,
    input_image: Image.Image | None,
    resolution: str,
    steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    seed: int,
    use_4bit: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    task = "edit" if input_image is not None else "generate"
    log.info("─" * 60)
    log.info("Task: %s | Model: %s", task, model_name)
    log.info("Prompt: %s", prompt[:80] + ("..." if len(prompt) > 80 else ""))
    log.info("Steps: %d | Guidance: %.1f | Seed: %d", steps, guidance_scale, seed)

    pipe = _load_pipeline(model_name, use_4bit=use_4bit)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    width, height = _parse_resolution(resolution)
    if input_image is not None:
        width, height = input_image.size
    log.info("Resolution: %d × %d", width, height)

    kwargs = {
        "prompt": prompt,
        "image": input_image,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "generator": generator,
    }

    config = MODEL_REGISTRY[model_name]
    if config["supports_cfg"]:
        kwargs["guidance_scale"] = guidance_scale
        kwargs["image_guidance_scale"] = image_guidance_scale

    t0 = time.perf_counter()
    result = pipe(**kwargs).images[0]
    elapsed = time.perf_counter() - t0

    log.info("Generated in %.2fs (%.2f steps/s)", elapsed, steps / elapsed)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        log.info("Peak VRAM: %.2f GB", peak)
        torch.cuda.reset_peak_memory_stats()

    if result.size != (width, height):
        result = result.resize((width, height), resample=Image.LANCZOS)

    return result


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
                        value=20,
                        step=1,
                        label="Inference Steps",
                    )
                    guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=3.5,
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
                        value=42,
                        label="Seed",
                        precision=0,
                    )
                    use_4bit_checkbox = gr.Checkbox(
                        value=True,
                        label="Quantize Text Encoder (8-bit, saves ~2 GB VRAM)",
                    )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Result")

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
            use_4bit_checkbox,
        ]

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[steps_slider, guidance_slider, guidance_slider, img_guidance_slider],
        )

        # Button click
        generate_btn.click(
            fn=generate,
            inputs=all_inputs,
            outputs=[output_image],
        )

        # Ctrl+Enter / submit from the prompt textbox
        prompt_input.submit(
            fn=generate,
            inputs=all_inputs,
            outputs=[output_image],
        )

        gr.Examples(
            examples=[
                [
                    list(MODEL_REGISTRY.keys())[0],
                    "A close-up of a fire-breathing dragon, cinematic shot",
                    None,
                    "832 × 1216 (2:3)",
                    20,
                    3.5,
                    1.0,
                    123,
                    True,
                ],
                [
                    list(MODEL_REGISTRY.keys())[1],
                    "A portrait of a young woman with flowers",
                    None,
                    "1024 × 1024 (1:1)",
                    4,
                    1.0,
                    1.0,
                    42,
                    True,
                ],
            ],
            inputs=all_inputs,
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DreamLite Gradio App")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7863, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    log.info("Starting DreamLite app on %s:%d", args.host, args.port)
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft())
