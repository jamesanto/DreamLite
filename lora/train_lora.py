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
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader, Dataset 

from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

# 导入你的核心组件
from dreamlite import DreamLitePipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for DreamLite")
    parser.add_argument("--model_id", type=str, default="models/DreamLite-base")
    parser.add_argument("--output_dir", type=str, default="./output/output_lora")
    parser.add_argument("--rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    # parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
    )
    
    # 2. Load DreamLite Pipeline
    pipe = DreamLitePipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    # Frozen other modules
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. LoRA (Based on PEFT)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
    )
    unet = get_peft_model(unet, lora_config)
    
    # print
    unet.print_trainable_parameters()

    # 4. configure optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    
    # 5. prepare dataloader
    # =======================================================
    # TODO: finish DataLoader
    # dataset = MyDataset(args.dataset_path, ...)
    # dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # =======================================================

    # 6. Accelerator
    # unet, optimizer = accelerator.prepare(unet, optimizer)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    vae.to(accelerator.device, dtype=torch.bfloat16)
    text_encoder.to(accelerator.device, dtype=torch.bfloat16)

    # 7. 开始训练
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    
    unet.train()
    
    while global_step < args.max_train_steps:
        # =======================================================
        # TODO: get data from DataLoader
        # for batch in dataloader:
        #     images = batch["pixel_values"]
        #     prompts = batch["text"]
        # =======================================================
        
        with accelerator.accumulate(unet):
            # 1. encode Latents (Ground Truth x_0)
            latents = vae.encode(images).latents
            latents = latents * vae.config.scaling_factor

            # 2. noise and timestep
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # 3. Add noise to Latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. Encode Prompt
            prompt_embeds, text_attention_mask = pipe.encode_prompt(
                mode="generate",
                prompts=prompts,
                device=accelerator.device,
                dtype=torch.bfloat16,
            )

            # 5. Time IDs, Image Latents
            # Generate mode, condition image = 0
            cond_img_in = torch.zeros_like(noisy_latents)
            model_input = torch.cat([noisy_latents, cond_img_in], dim=3) # In-context Concat
            
            add_time_ids = torch.tensor([[1024, 1024]], dtype=torch.bfloat16, device=accelerator.device).repeat(bsz, 1)

            # 6. UNet Predict Noise
            noise_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=text_attention_mask,
                added_cond_kwargs={"time_ids": add_time_ids},
                return_dict=False,
            )[0]
            
            noise_pred = noise_pred[..., :latents.shape[-1]]

            # 7. Loss (Flow Matching, MSE)
            target = noise 
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            # 8. backward and update params
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()

        # update
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix({"loss": loss.item()})

    accelerator.wait_for_everyone()
    
    # 8. Save LoRA weights
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        os.makedirs(args.output_dir, exist_ok=True)
        unet.save_pretrained(args.output_dir)
        print(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()