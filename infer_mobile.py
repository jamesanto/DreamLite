import os
import io
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers.utils import load_image
from modules.model_utils import load_model

# import bytenn_torch

warnings.filterwarnings("ignore")



def parse_args(): 
    parser = argparse.ArgumentParser(description="DreamLite Inference Script, 4 Step, w/o CFG & IMG_CFG")
    
    # Model & Structure
    parser.add_argument("--model_path", type=str, default="/mnt/bn/mobile-hl/projects/DreamLite/models/20-40000")
        
    # Inference Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bucket", type=int, default=0, choices=[0, 1, 2, 54, 765])
    parser.add_argument("--weight_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="a naked man")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
        
    return parser.parse_args()

# --- Main Execution ---

def main():
    args = parse_args()
        
    weight_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.weight_dtype]

    # 2. Load Model
    pipeline = load_model(
        args.model_path,
        device=args.device,
        dtype=weight_dtype,
    )

    # 3. Setup Data
    prompt = args.prompt
    input_image = load_image(args.image_path) if args.image_path else None
    if input_image is not None:
        width, height = input_image.size
    else:
        width, height = args.height, args.width

    # 4. Setup Output Directory
    image = pipeline(
        prompt=prompt,
        image=input_image,
        height=height, # Pipeline expects H, W order usually
        width=width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    
    image.save(f"{prompt.replace(' ', '_')}.png")


if __name__ == "__main__":
    main()


"""
source /mnt/bn/humangen-hl2/kailai/miniconda3/bin/activate 
conda activate dreamlite
Edit task: python infer_mobile.py --image_path "a_photo_of_a_cat.png"  --prompt "let the cat wear sunglasses" 
Generate task: python infer_mobile.py --prompt "a photo of a cat"
"""