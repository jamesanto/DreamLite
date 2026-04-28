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

import os
import json
import torch
from accelerate import Accelerator
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers.utils import load_image

# 用于 GenEval 的拼图工具
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from dreamlite import DreamLitePipeline

warnings.filterwarnings("ignore")

class GenEvalDataset(Dataset):
    def __init__(self, prompt_path: str, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # 1. 读取jsonl
        self.data = []
        with open(prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                self.data.append(obj)

        # 2. 按照idx保存
        for idx, obj in enumerate(self.data):
            out_dir = os.path.join(self.save_dir, f"{idx:05d}")
            os.makedirs(out_dir, exist_ok=True)
            meta_path = os.path.join(out_dir, "metadata.jsonl")
            with open(meta_path, "w", encoding="utf-8") as mf:
                mf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        prompt = obj.get("prompt", "")
        tag = obj.get("tag", "")
        
        return {
            "idx": idx,
            "prompt": prompt,
            "tag": tag
        }



def parse_args():
    parser = argparse.ArgumentParser(description="DreamLite Inference Script")
    
    # Model & Structure
    parser.add_argument("--model_path", type=str, default="models/DreamLite-base")
        
    # Inference Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bucket", type=int, default=0, choices=[0, 1, 2, 54, 765])
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default="./output/benchmark/geneval_output")
    parser.add_argument("--geneval_json", type=str, default="YOUR_GENEVAL/evaluation_metadata.jsonl")
        
    return parser.parse_args()

# --- Main Execution ---

def main():
    args = parse_args()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    accelerator = Accelerator()
        
    weight_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.weight_dtype]

    # 2. Load Model
    pipeline = DreamLitePipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
    ).to(args.device)
    pipeline = accelerator.prepare(pipeline)

    # 3. Setup Data
    dataset = GenEvalDataset(
        prompt_path=args.geneval_json, 
        save_dir=args.save_dir,
    )
    loader = accelerator.prepare(DataLoader(dataset))
    width, height = args.width, args.height

    # 4. inference
    for batch in tqdm(loader):
        idx = int(batch['idx'][0].item())
        prompt = batch['prompt'][0]

        output_folder = os.path.join(save_dir, f"{idx:05d}")
        output_sample = os.path.join(output_folder, "samples")
        os.makedirs(output_sample, exist_ok=True)

        num_samples = 4
        all_samples = []
        grid_path = os.path.join(save_dir, f"{idx:05d}_grid.jpg")
        if os.path.exists(grid_path):
            continue

        print(f"##### generating {idx} {prompt} #####")
        for i in range(num_samples):
            seed = 2024 + i
            image = pipeline(
                prompt=prompt,
                height=height, # Pipeline expects H, W order usually
                width=width,
                guidance_scale=7.5,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                bucket=args.bucket,
            ).images[0]

            image.save(os.path.join(output_sample, f"{i:05d}.png"))
            all_samples.append(torch.stack([ToTensor()(image)], 0))

        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=4)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        grid.save(grid_path)


if __name__ == "__main__":
    main()
