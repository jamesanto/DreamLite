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

class ImgEditBenchDataset(Dataset):
    def __init__(
        self,
        json_path,
        img_root,
        img_size = 1024,
        img_tf = None,
        keep_edit_type = False,
    ):
        self.img_root = img_root
        self.img_size = img_size
        self.keep_edit_type = keep_edit_type

        with open(json_path, "r", encoding="utf-8") as f:
            jdict = json.load(f)

        # 转成列表方便按 idx 访问；保留键可追溯原始 id
        self.records = [
            {"key": k, **v}
            for k, v in jdict.items()
            if all(x in v for x in ("id", "prompt", "edit_type"))
        ]
    
    def __len__(self) -> int:
        return len(self.records)
        
    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = os.path.join(self.img_root, rec["id"])

        sample = {
            "image_path":  image_path,
            "prompt": rec["prompt"],
            "key":    int(rec["key"]),  
        }
        if self.keep_edit_type:
            sample["edit_type"] = rec["edit_type"]

        return sample


def load_dataloader(img_size, json_path, img_root):
    def collate_single(batch):
        return batch[0]

    imgedit_ds = ImgEditBenchDataset(
        json_path=json_path,
        img_root=img_root,
        img_size=img_size,
    )
    loader = DataLoader(imgedit_ds, collate_fn=collate_single)

    return loader


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
    parser.add_argument("--save_dir", type=str, default="./output/benchmark/imgedit_output")
    parser.add_argument("--json_path", type=str, default="YOUR_IMGEDIT_PATH/ImgEdit/Benchmark/Basic/basic_edit.json")
    parser.add_argument("--img_root", type=str, default="YOUR_IMGEDIT_IMAGES_PATH/ImgEdit/Benchmark/singleturn")
        
    return parser.parse_args()

# --- Main Execution ---

def main():
    args = parse_args()
    accelerator = Accelerator()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
        
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
    imgedit_dataloader = load_dataloader(img_size=args.width, json_path=args.json_path, img_root=args.img_root)
    eval_dl = accelerator.prepare(imgedit_dataloader) 

    width, height = args.width, args.height

    # 4. inference
    for batch in tqdm(eval_dl):
        img_path = batch["image_path"]
        key = batch["key"]
        prompt = batch["prompt"]

        save_path = os.path.join(save_dir, f'{key}.png')
        if os.path.exists(save_path):
            continue
            
        init_img = load_image(img_path)
        ori_size = init_img.size
        input_image = init_img.resize((width, height), Image.Resampling.LANCZOS)

        image = pipeline(
            prompt=prompt,
            image=input_image,
            height=height, # Pipeline expects H, W order usually
            width=width,
            guidance_scale=3.5,
            image_guidance_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(42),
            bucket=args.bucket,
        ).images[0]

        image = image.resize(ori_size, Image.Resampling.LANCZOS)
        image.save(save_path)

if __name__ == "__main__":
    main()
