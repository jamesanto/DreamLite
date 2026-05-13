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

import time as _time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderTiny
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...models.unets.unet_2d_condition_mobile import DreamLiteUNetModel
from .optimize import compile_unet, enable_fast_attention, move_to_device, offload_to_cpu

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

# ==========================================
# Constant Definitions
# ==========================================
TARGET_BUCKETS_V54 = [
    [1248, 832],
    [1024, 1024],
    [896, 1184],
    [832, 1248],
    [1376, 768],
    [1184, 896],
    [928, 1120],
    [864, 1216],
    [1216, 864],
    [1312, 800],
    [768, 1376],
    [1280, 832],
    [1152, 896],
    [1344, 768],
    [1120, 928],
    [1408, 736],
    [1440, 736],
]

TARGET_BUCKETS_V765 = [
    [1248, 832],
    [1024, 1024],
    [896, 1152],
    [1248, 832],
    [960, 1088],
    [1088, 960],
    [1152, 896],
    [832, 1248],
    [832, 1248],
    [1312, 800],
    [800, 1312],
    [1344, 768],
    [768, 1344],
    [1440, 736],
    [736, 1440],
    [1472, 704],
    [704, 1472],
    [1600, 672],
    [672, 1568],
    [1184, 896],
]


# ==========================================
# Helper Functions
# ==========================================
def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
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


def _get_closest_bucket(buckets: List[List[int]], w: int, h: int) -> Tuple[int, int]:
    target_ar = w / h
    best_bucket = min(buckets, key=lambda b: abs((b[0] / b[1]) - target_ar))
    best_bucket = [int(x * 2) for x in best_bucket]
    return tuple(best_bucket)


# ==========================================
# Pipeline Class
# ==========================================
class DreamLitePipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    def __init__(
        self,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        processor: Qwen3VLProcessor,
        vae: AutoencoderTiny,
        unet: DreamLiteUNetModel,
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

        # 安全计算 VAE scale factor，避免读取非标准配置报错，默认通常为 8 (2^3)
        if hasattr(self.vae.config, "encoder_block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        self._offload_text_encoder = False
        self._unet_compiled = False

    def optimize(
        self,
        offload_text_encoder: bool = True,
        compile_unet_model: bool = True,
        fuse_qkv: bool = True,
        enable_vae_tiling: bool = True,
    ) -> "DreamLitePipeline":
        """
        Apply speed/VRAM optimizations for low-VRAM GPUs (e.g. RTX 2060 6GB).

        Args:
            offload_text_encoder: Move text encoder to CPU after encoding to free VRAM
                for the UNet denoising loop.
            compile_unet_model: Apply torch.compile to UNet for kernel fusion.
            fuse_qkv: Fuse Q/K/V attention projections into a single matmul.
            enable_vae_tiling: Enable tiled VAE decoding to prevent OOM during decode.

        Returns:
            self (for method chaining)
        """
        self._offload_text_encoder = offload_text_encoder

        # Enable optimal CUDA attention backends
        enable_fast_attention()

        if fuse_qkv:
            try:
                self.unet.fuse_qkv_projections()
                logger.info("Fused QKV projections in UNet.")
            except (AttributeError, RuntimeError) as e:
                logger.warning("QKV fusion not supported for this model: %s", e)

        if compile_unet_model and not self._unet_compiled:
            self.unet = compile_unet(self.unet)
            self._unet_compiled = True

        if enable_vae_tiling and hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
            logger.info("VAE tiling enabled.")

        return self

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """Extract valid hidden states based on attention mask."""
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
        image: Optional[Image.Image] = None,
        max_sequence_length: int = 500,
        text_pad_embedding: Optional[torch.Tensor] = None,
    ):
        if mode == "edit":
            drop_idx = 64
            template = (
                "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
                "texture, objects, background), then explain how the user's text instruction should alter "
                "or modify the image. Generate a new image that meets the user's requirements while maintaining "
                "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            )

            txts = [template.format(p) for p in prompts]
            images = [image.resize((512, 512), Image.Resampling.LANCZOS)] * len(prompts)

            tk_out = self.processor(text=txts, images=images, padding=True, return_tensors="pt").to(device)

            outputs = self.text_encoder(
                input_ids=tk_out.input_ids,
                attention_mask=tk_out.attention_mask,
                pixel_values=tk_out.pixel_values,
                image_grid_thw=tk_out.image_grid_thw,
                # mm_token_type_ids=tk_out.mm_token_type_ids,
                output_hidden_states=True,
            )

        elif mode == "generate":
            drop_idx = 34
            template = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
                "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )

            txts = [template.format(p) for p in prompts]
            tk_out = self.tokenizer(text=txts, padding=True, return_tensors="pt").to(device)

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

        prompt_embeds = pad_sequence(split_hidden_states, batch_first=True, padding_value=0).to(
            dtype=dtype, device=device
        )

        B, L, _ = prompt_embeds.shape
        prompt_embeds_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, seq in enumerate(split_hidden_states):
            prompt_embeds_mask[i, : seq.shape[0]] = 1

        # Apply text_pad_embedding if provided
        if text_pad_embedding is not None:
            pad_emb = text_pad_embedding.to(dtype=dtype, device=device)
            if pad_emb.ndim == 1:
                pad_emb = pad_emb.unsqueeze(0).unsqueeze(0)
            elif pad_emb.ndim == 2:
                pad_emb = pad_emb.unsqueeze(0)

            mask_expanded = prompt_embeds_mask.unsqueeze(-1).to(dtype=dtype)
            prompt_embeds = prompt_embeds * mask_expanded + pad_emb * (1 - mask_expanded)

        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(f"`image` must be of type `torch.Tensor`, `PIL.Image.Image` or `list`, got {type(image)}")

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 200,
        text_pad_embedding: Optional[torch.Tensor] = None,
        bucket: int = 0,
        callback_on_step_end: Optional[callable] = None,
        interrupt_flag: Optional[callable] = None,
    ):
        # 1. Init pipeline parameters
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        task = "generate" if image is None else "edit"
        device = self._execution_device
        dtype = self.text_encoder.dtype
        batch_size = 1  # Note: Currently forced to batch_size 1 for this pipeline
        negative_prompt = negative_prompt or ""

        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        _use_cuda = torch.cuda.is_available() and str(device) != "cpu"

        def _sync():
            if _use_cuda:
                torch.cuda.synchronize()

        _profile = {}
        _t_total_start = _time.perf_counter()

        # 2. Prepare Dimensions (Buckets)
        if image is not None:  # edit task, resize to certain bucket
            if bucket == -1:
                pass  # caller already set width/height to the optimal resolution
            elif bucket == 0:
                height = width = 1024
            elif bucket == 1:
                height = width = 2048
            elif bucket == 54:
                width, height = _get_closest_bucket(TARGET_BUCKETS_V54, width, height)
            elif bucket == 765:
                width, height = _get_closest_bucket(TARGET_BUCKETS_V765, width, height)
            else:
                height, width = height, width

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
        _sync()
        _t0 = _time.perf_counter()

        if task == "generate":
            prompt_str = f"[Generate]: {prompt}"
            prompt_embeds, text_attention_mask = self.encode_prompt(
                mode="generate",
                prompts=[negative_prompt, prompt_str],
                device=device,
                dtype=dtype,
                text_pad_embedding=text_pad_embedding,
            )
            image_latents = torch.zeros_like(latents)

        else:
            prompt_str = f"[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has {prompt}"
            prompt_embeds, text_attention_mask = self.encode_prompt(
                mode="edit",
                prompts=[negative_prompt, negative_prompt, prompt_str],
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
            uncond_image_latents = torch.zeros_like(latents)

        _sync()
        _profile["text_encode"] = _time.perf_counter() - _t0

        # 6b. Offload text encoder to CPU to free VRAM for the UNet loop
        if self._offload_text_encoder:
            _t0 = _time.perf_counter()
            offload_to_cpu(self.text_encoder)
            _profile["text_offload"] = _time.perf_counter() - _t0

        # 7. Denoising Loop
        _sync()
        if _use_cuda:
            _free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            _alloc = torch.cuda.memory_allocated() / (1024**3)
            logger.info(
                "UNet loop start — VRAM: %.2f GB allocated, %.2f GB free (batch=%d)",
                _alloc, _free_vram, 2 if task == "generate" else 3,
            )
        _t_unet_start = _time.perf_counter()
        _step_times = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                _t_step = _time.perf_counter()

                # Expand latents for classifier-free guidance
                if task == "generate":
                    latents_in = torch.cat([latents] * 2)
                    cond_img_in = torch.cat([image_latents] * 2)
                    model_input = torch.cat([latents_in, cond_img_in], dim=3)
                    time_ids_in = torch.cat([add_time_ids] * 2)

                elif task == "edit":
                    latents_in = torch.cat([latents] * 3)
                    cond_img_in = torch.cat([uncond_image_latents, image_latents, image_latents])
                    model_input = torch.cat([latents_in, cond_img_in], dim=3)
                    time_ids_in = torch.cat([add_time_ids] * 3)

                # UNet Forward
                noise_pred = self.unet(
                    model_input,
                    timestep=t.expand(model_input.shape[0]).to(latents.dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=text_attention_mask,
                    added_cond_kwargs={"time_ids": time_ids_in},
                    return_dict=False,
                )[0]

                # Classifier-Free Guidance
                noise_pred = noise_pred[..., : latents.shape[-1]]
                if task == "generate":
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_cond - noise_pred_uncond)
                elif task == "edit":
                    noise_pred_uncond, noise_pred_image, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self._guidance_scale * (noise_pred_text - noise_pred_image)
                        + self._image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # Scheduler Step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                _sync()
                _step_times.append(_time.perf_counter() - _t_step)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(i + 1, num_inference_steps, _step_times[-1])

                if interrupt_flag is not None and interrupt_flag():
                    logger.info("Generation interrupted at step %d/%d", i + 1, num_inference_steps)
                    break

                if XLA_AVAILABLE:
                    xm.mark_step()

        _sync()
        _profile["unet_total"] = _time.perf_counter() - _t_unet_start
        _profile["unet_steps"] = num_inference_steps
        _profile["unet_per_step_avg"] = _profile["unet_total"] / num_inference_steps
        _profile["unet_per_step_first"] = _step_times[0] if _step_times else 0
        _profile["unet_per_step_last"] = _step_times[-1] if _step_times else 0

        # 8. Decode Latents
        _sync()
        _t0 = _time.perf_counter()

        if output_type == "latent":
            image_out = latents
        else:
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
            latents = (latents / self.vae.config.scaling_factor) + shift_factor
            image_out = self.vae.decode(latents, return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        _sync()
        _profile["vae_decode"] = _time.perf_counter() - _t0

        # 8b. Restore text encoder to GPU for next call if it was offloaded
        if self._offload_text_encoder:
            _t0 = _time.perf_counter()
            move_to_device(self.text_encoder, device, dtype)
            _profile["text_reload"] = _time.perf_counter() - _t0

        _profile["total"] = _time.perf_counter() - _t_total_start

        # 9. Print profiling summary
        if _use_cuda:
            _vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            _vram_current = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
        else:
            _vram_peak = _vram_current = 0.0

        _cfg_batch = 2 if task == "generate" else 3
        logger.info(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║              DreamLite Profiling Summary                 ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║ Task: %-10s  Resolution: %d×%d  CFG batch: %d     ║\n"
            "║ Steps: %d         Dtype: %-10s Device: %-10s  ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║ Phase                          Time                      ║\n"
            "║ ─────────────────────────────  ─────────                 ║\n"
            "║ Text Encode (Qwen3-VL)         %6.2fs                    ║\n"
            "║ Text Offload → CPU             %6.2fs                    ║\n"
            "║ UNet Loop (%2d steps)            %6.2fs                    ║\n"
            "║   ├─ First step                %6.3fs                    ║\n"
            "║   ├─ Last step                 %6.3fs                    ║\n"
            "║   └─ Avg per step              %6.3fs                    ║\n"
            "║ VAE Decode                     %6.2fs                    ║\n"
            "║ Text Reload → GPU              %6.2fs                    ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║ TOTAL                          %6.2fs                    ║\n"
            "║ Peak VRAM: %.2f GB  Current: %.2f GB                  ║\n"
            "╚══════════════════════════════════════════════════════════╝",
            task,
            width,
            height,
            _cfg_batch,
            num_inference_steps,
            str(dtype).split(".")[-1],
            str(device),
            _profile.get("text_encode", 0),
            _profile.get("text_offload", 0),
            num_inference_steps,
            _profile.get("unet_total", 0),
            _profile.get("unet_per_step_first", 0),
            _profile.get("unet_per_step_last", 0),
            _profile.get("unet_per_step_avg", 0),
            _profile.get("vae_decode", 0),
            _profile.get("text_reload", 0),
            _profile.get("total", 0),
            _vram_peak,
            _vram_current,
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)

        return FluxPipelineOutput(images=image_out)
