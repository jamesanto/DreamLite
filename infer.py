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
    _BNB_AVAILABLE,
    get_4bit_quantization_config,
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
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="a dog running on the grass")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.0)

    # Optimization flags
    parser.add_argument(
        "--quantize_4bit", action="store_true", default=True, help="Load text encoder in 4-bit (requires bitsandbytes)"
    )
    parser.add_argument("--no_quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument(
        "--no_optimize", action="store_true", help="Disable all pipeline optimizations (compile, fuse, offload)"
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

    use_4bit = args.quantize_4bit and not args.no_quantize
    if use_4bit and _BNB_AVAILABLE:
        from transformers import Qwen3VLForConditionalGeneration

        print("Loading text encoder with 4-bit NF4 quantization...")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            subfolder="text_encoder",
            quantization_config=get_4bit_quantization_config(compute_dtype=weight_dtype),
            device_map="auto",
        )
        load_kwargs["text_encoder"] = text_encoder
    elif use_4bit and not _BNB_AVAILABLE:
        print("Warning: bitsandbytes not installed; loading without quantization.")

    print(f"Loading diffusers pipeline from: {args.model_id}")
    pipeline = DreamLitePipeline.from_pretrained(args.model_id, **load_kwargs)

    # Move components to device: quantized text encoder already on CUDA via device_map
    if use_4bit and _BNB_AVAILABLE:
        pipeline.unet.to(args.device, dtype=weight_dtype)
        pipeline.vae.to(args.device, dtype=weight_dtype)
    else:
        pipeline = pipeline.to(args.device)

    # Apply speed optimizations
    if not args.no_optimize:
        pipeline.optimize(
            offload_text_encoder=True,
            compile_unet_model=True,
            fuse_qkv=True,
            enable_vae_tiling=True,
        )
        print("Pipeline optimized (offload + compile + fuse_qkv + vae_tiling).")

    # Setup Data
    prompt = args.prompt
    input_image = load_image(args.image_path) if args.image_path else None
    if input_image is not None:
        width, height = input_image.size
    else:
        width, height = args.width, args.height

    print("Generating image...")
    image = pipeline(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image=input_image,
        height=height,
        width=width,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    out_path = f"{prompt.replace(' ', '_')}.png"
    image.save(out_path)
    print(f"Image saved to {out_path}")


if __name__ == "__main__":
    main()
