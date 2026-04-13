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

import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Qwen3VLForConditionalGeneration, 
    AutoProcessor
)
from diffusers import (
    AutoencoderKL, 
    DreamLiteUNetModel, 
    UNet2DConditionModel, 
    FlowMatchEulerDiscreteScheduler
)
from diffusers.models import AutoencoderTiny
from modules.pipeline_utils_mobile import DreamLiteMobilePipeline
from modules.pipeline_utils import DreamLitePipeline

def load_model(
    model_path, device="cuda", dtype="bfloat16", mode="mobile"
):
    print(f'[INFO] Loading Checkpoint: {model_path}')

    # text-encoder
    text_encoder_path = "models/Qwen3-VL-2B-Instruct"
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(text_encoder_path).to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    processor = AutoProcessor.from_pretrained(text_encoder_path)

    # VAE
    vae_path = "models/taesdxl"
    vae = AutoencoderTiny.from_pretrained(vae_path).to(device, dtype=dtype)

    # UNet
    unet_path = "./modules/dreamlite.json"
    if model_path.endswith(".pt"):
        with open(unet_path, "r") as f:
            unet_config = json.load(f)
        unet = DreamLiteUNetModel(**unet_config).to(device, dtype=dtype)
        unet_state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["module"]
        load_result = unet.load_state_dict(unet_state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"Key Mismatches: Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}")
    else:
        unet = DreamLiteUNetModel.from_pretrained(model_path).to(device, dtype=dtype)

      
    # Scheduler & Extras
    scheduler_path = "models"
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path, subfolder="scheduler")
    
    # Build Pipeline
    if mode == 'mobile':
        pipe = DreamLiteMobilePipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            vae=vae,
            unet=unet,
            scheduler=noise_scheduler,
        )
        pipe.ckpt_path = unet_path
    else:
        pipe = DreamLitePipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            vae=vae,
            unet=unet,
            scheduler=noise_scheduler,
        )
        
    return pipe

