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

from dreamlite import DreamLiteMobilePipeline 
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="DreamLite-Mobile Inference Script")
    
    # Model & Structure
    parser.add_argument("--model_id", type=str, default="models/DreamLite-mobile")
        
    # Inference Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="a portrait of a young woman with flowers")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
        
    return parser.parse_args()

def main():
    args = parse_args()
        
    weight_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.weight_dtype]

    print(f"Loading diffusers pipeline from: {args.model_id}")
    
    pipeline = DreamLiteMobilePipeline.from_pretrained(
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
