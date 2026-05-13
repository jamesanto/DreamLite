# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import warnings

import torch
from diffusers.utils import load_image
from PIL import Image

from dreamlite import DreamLitePipeline
from dreamlite.pipelines.dreamlite.optimize import (
    get_optimal_dtype,
    is_turing_gpu,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="DreamLite Inference Script")

    # Model & Structure
    parser.add_argument("--model_id", type=str, default="models/DreamLite-base")

    # Inference Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--prompt", type=str, default="a dog running on the grass")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.0)

    # Optimization flags
    parser.add_argument(
        "--no_optimize", action="store_true", help="Disable all pipeline optimizations (compile, fuse, offload)"
    )
    parser.add_argument("--no_compile", action="store_true", help="Disable CUDA graph acceleration (eager mode)")
    parser.add_argument("--use_inductor", action="store_true", help="Use torch.compile inductor backend instead of CUDA graphs")
    parser.add_argument(
        "--vae_tiling", action="store_true", help="Enable VAE tiling (prevents OOM at high resolutions)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve dtype: auto-detect optimal dtype for the GPU
    if args.weight_dtype == "auto":
        weight_dtype = get_optimal_dtype()
        arch_name = "Turing (fp16)" if is_turing_gpu() else "Ampere+ (bf16)"
        print(f"Auto-detected GPU architecture: {arch_name}, using {weight_dtype}")
    else:
        weight_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.weight_dtype]

    # Build loading kwargs
    load_kwargs = {"torch_dtype": weight_dtype}

    print(f"Loading diffusers pipeline from: {args.model_id} (fp16 text encoder, CPU offload)")
    pipeline = DreamLitePipeline.from_pretrained(args.model_id, **load_kwargs)
    pipeline = pipeline.to(args.device)

    # Apply speed optimizations
    if not args.no_optimize:
        use_cuda_graph = not args.no_compile and not args.use_inductor
        use_inductor = args.use_inductor and not args.no_compile
        pipeline.optimize(
            offload_text_encoder=True,
            compile_unet_model=use_inductor,
            use_cuda_graph=use_cuda_graph,
            fuse_qkv=False,
            enable_vae_tiling=args.vae_tiling,
        )
        accel = "CUDA graphs" if use_cuda_graph else ("inductor" if use_inductor else "none")
        print(
            f"Pipeline optimized (SDPA + {accel}{' + vae_tiling' if args.vae_tiling else ''})."
        )

    # Setup Data
    prompt = args.prompt
    input_image = load_image(args.image_path) if args.image_path else None

    EDIT_BUCKETS = [
        (1024, 1024), (1152, 896), (896, 1152),
        (1216, 832), (832, 1216), (1344, 768), (768, 1344),
    ]

    crop_box = None
    if input_image is not None:
        img_w, img_h = input_image.size
        target_ar = img_w / img_h
        bucket_w, bucket_h = min(EDIT_BUCKETS, key=lambda b: abs((b[0] / b[1]) - target_ar))

        img_long = max(img_w, img_h)
        bucket_long = max(bucket_w, bucket_h)
        scale = bucket_long / img_long
        new_w = min(round(img_w * scale), bucket_w)
        new_h = min(round(img_h * scale), bucket_h)

        resized = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        pad_left = (bucket_w - new_w) // 2
        pad_top = (bucket_h - new_h) // 2
        padded = Image.new("RGB", (bucket_w, bucket_h), (128, 128, 128))
        padded.paste(resized, (pad_left, pad_top))

        crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
        input_image = padded
        width, height = bucket_w, bucket_h
        print(f"Edit mode: input {img_w}×{img_h} → resize {new_w}×{new_h} → pad to {width}×{height}")
    else:
        width, height = args.width, args.height

    print("Generating image...")

    def cli_step_callback(step: int, total: int, step_time: float):
        pct = step * 100 // total
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {step}/{total} steps ({step_time:.2f}s/step)", end="", flush=True)
        if step == total:
            print()

    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "image": input_image,
        "height": height,
        "width": width,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator("cpu").manual_seed(42),
        "callback_on_step_end": cli_step_callback,
    }
    if crop_box is not None:
        call_kwargs["bucket"] = -1

    image = pipeline(**call_kwargs).images[0]

    if crop_box is not None:
        image = image.crop(crop_box)
        print(f"Cropped padding: {width}×{height} → {image.size[0]}×{image.size[1]}")

    out_path = f"{prompt.replace(' ', '_')}.png"
    image.save(out_path)
    print(f"Image saved to {out_path}")


if __name__ == "__main__":
    main()
