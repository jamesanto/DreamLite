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

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents, deprecate
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.unets import UNet2DConditionModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> pipe = NuwaPipeline.from_pretrained(...)
        >>> image = pipe(prompt="A cat", num_inference_steps=28).images[0]
        ```
"""

# 预定义的宽高比 Bucket (Base 1024x1024)
TARGET_BUCKETS_V54 = [
    [1248, 832], [1024, 1024], [896, 1184], [832, 1248],
    [1376, 768], [1184, 896], [928, 1120], [864, 1216],
    [1216, 864], [1312, 800], [768, 1376], [1280, 832],
    [1152, 896], [1344, 768], [1120, 928], [1408, 736], [1440, 736]
]
TARGET_BUCKETS_V765 = [
    [1248, 832], [1024, 1024], [896, 1152], [1248, 832], [960, 1088], 
    [1088, 960], [1152, 896], [832, 1248], [832, 1248], [1312, 800], 
    [800, 1312], [1344, 768], [768, 1344], [1440, 736], [736, 1440], 
    [1472, 704], [704, 1472], [1600, 672], [672, 1568], [1184, 896]
]

# --- Helper Functions ---

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _get_closest_bucket(buckets, w, h):
    target_ar = w / h
    best_bucket = min(buckets, key=lambda b: abs((b[0]/b[1]) - target_ar))
    best_bucket = [int(x * 2) for x in best_bucket]
    return tuple(best_bucket)


# --- Pipeline Class ---

class NuwaPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    def __init__(
        self,
        text_encoder: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        # 优化：直接使用 self.vae.config 而不是 getattr
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        """静态方法：根据 mask 提取有效的 hidden states"""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def encode_prompt(
        self,
        mode: str,
        prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
        image: Optional[torch.Tensor] = None,
        max_sequence_length: int = 500,
        text_pad_embedding: Optional[torch.Tensor] = None,
    ):
        # mode = 'generate'  # drop the image token
        if mode == "edit":
            drop_idx = 64 
            template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            
            txts = [template.format(p) for p in prompts]
            images = [image.resize((224, 224), Image.Resampling.LANCZOS)] * len(prompts)
            
            tk_out = self.processor(
                text=txts, images=images, padding=True, return_tensors="pt",
            ).to(device)

            outputs = self.text_encoder(
                input_ids=tk_out.input_ids,
                attention_mask=tk_out.attention_mask,
                pixel_values=tk_out.pixel_values,
                image_grid_thw=tk_out.image_grid_thw,
                output_hidden_states=True
            )
        elif mode == 'generate':
            drop_idx = 34
            template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            
            txts = [template.format(p) for p in prompts]
            tk_out = self.tokenizer(
                text=txts, padding=True, return_tensors="pt",
            ).to(device)

            outputs = self.text_encoder(
                input_ids=tk_out.input_ids,
                attention_mask=tk_out.attention_mask,
                output_hidden_states=True,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, tk_out.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        
        prompt_embeds = pad_sequence(split_hidden_states, batch_first=True, padding_value=0).to(dtype=dtype, device=device)

        B, L, _ = prompt_embeds.shape
        prompt_embeds_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, seq in enumerate(split_hidden_states):
            prompt_embeds_mask[i, :seq.shape[0]] = 1
        
        if text_pad_embedding is not None:
            # 确保维度匹配 [1, 1, D] 以便广播
            pad_emb = text_pad_embedding.to(dtype=dtype, device=device)
            if pad_emb.ndim == 1:
                pad_emb = pad_emb.unsqueeze(0).unsqueeze(0)
            elif pad_emb.ndim == 2:
                pad_emb = pad_emb.unsqueeze(0)
            
            # Mask 扩展: [B, L, 1]
            mask_expanded = prompt_embeds_mask.unsqueeze(-1).to(dtype=dtype)
            # 融合：Mask位置保留原值，非Mask位置使用 pad_emb
            prompt_embeds = prompt_embeds * mask_expanded + pad_emb * (1 - mask_expanded)

        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generator list length must match batch size.")

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_image_latents(
            self, image, dtype, device, generator=None
    ):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        image = image.to(device=device, dtype=dtype)
        batch_size = 1

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
        
        return image_latents

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 200,
        text_pad_embedding: Optional[torch.Tensor] = None,
        bucket: int = 0,
    ):
        # 1. Init Locals
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        task = "generate" if image is None else "edit"
        device = self._execution_device
        dtype = self.text_encoder.dtype
        batch_size = 1 # 强制 batch=1
        
        # 使用局部变量 sigmas
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas

        # 2. Prepare Dimensions (Buckets)
        if bucket == 1:
            height = width = 2048
        elif bucket == 54:
            height, width = _get_closest_bucket(TARGET_BUCKETS_V54, width, height)
        elif bucket == 765:
            height, width = _get_closest_bucket(TARGET_BUCKETS_V765, width, height)
        else:
            height = width = 1024

        # 3. Prepare Time IDs
        original_size = (width, height)
        add_time_ids = torch.tensor([list(original_size)], device=device, dtype=dtype)

        # 4. Prepare Noise Latents (x_t)
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )
        
        # 5. Prepare Timesteps
        image_seq_len = latents.shape[2] * latents.shape[3] // 4 
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 6. Prepare Conditions (Text & Image)
        if task == 'generate':
            # === Generate Mode ===
            prompt = "[Generate]: " + prompt
            prompt_embeds, text_attention_mask = self.encode_prompt(
                mode="generate", 
                prompts=[prompt], 
                device=device,
                dtype=dtype,
                text_pad_embedding=text_pad_embedding,
            )
            image_latents = torch.zeros_like(latents)
            # image_processed = self.image_processor.preprocess(Image.new("RGB", (width, height), (0, 0, 0)))
            # _, image_latents = self.prepare_image_latents(image_processed, batch_size=1, num_images_per_prompt=1, dtype=dtype, device=device,)
            
        else: 
            # === Edit Mode ===
            prompt = "[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has " + prompt
            prompt_embeds, text_attention_mask = self.encode_prompt(
                mode="edit",
                prompts=[prompt],
                image=image,
                device=device,
                dtype=dtype,
            )
            image_processed = self.image_processor.preprocess(image.resize((width, height), Image.Resampling.LANCZOS))
            image_latents = self.prepare_image_latents(
                image_processed,
                dtype=dtype,
                device=device,
            )

        # 7. Denoising Loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # === Construct Batch Input ===
                if task == "generate":
                    latents_in = latents
                    cond_img_in = image_latents
                    model_input = torch.cat([latents_in, cond_img_in], dim=3)
                    # model_input = latents_in
                    time_ids_in = add_time_ids

                elif task == "edit":
                    latents_in = latents
                    cond_img_in = image_latents 
                    model_input = torch.cat([latents_in, cond_img_in], dim=3) 
                    time_ids_in = add_time_ids

                # === UNet Forward ===
                noise_pred = self.unet(
                    model_input,
                    timestep=t.expand(model_input.shape[0]).to(latents.dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=text_attention_mask,
                    added_cond_kwargs={"time_ids": time_ids_in},
                    return_dict=False, 
                )[0]

                # === Classifier-Free Guidance ===

                noise_pred = noise_pred[..., :latents.shape[-1]]

                # === Scheduler Step ===
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. Decode Latents
        if output_type == "latent":
            image = latents
        else:
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
            latents = (latents / self.vae.config.scaling_factor) + shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)