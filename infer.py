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
from PIL import Image
from diffusers.utils import load_image

from dreamlite import DreamLitePipeline 
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="DreamLite Inference Script")
    
    # Model & Structure
    # 这里不再指向具体的 .pt 文件，而是指向整个 diffusers 目录（或以后填入 Hugging Face 的 Repo ID，如 "Carlo/DreamLite"）
    parser.add_argument("--model_id", type=str, default="models/DreamLite-base")
        
    # Inference Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--prompt", type=str, default="a dog running on the grass")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.0)
        
    return parser.parse_args()

def main():
    args = parse_args()
        
    weight_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.weight_dtype]

    print(f"Loading diffusers pipeline from: {args.model_id}")
    
    pipeline = DreamLitePipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
    ).to(args.device)
    
    # 3. Setup Data
    prompt = args.prompt
    input_image = load_image(args.image_path) if args.image_path else None
    if input_image is not None:
        width, height = input_image.size
    else:
        width, height = args.height, args.width

    print("Generating image...")
    image = pipeline(
        prompt=prompt,
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