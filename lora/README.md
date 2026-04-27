# 🎛️ DreamLite LoRA Fine-Tuning Guide

This document provides instructions for training and performing inference with Low-Rank Adaptation (LoRA) on the DreamLite model. LoRA allows you to efficiently fine-tune DreamLite for specific artistic styles, subjects, or domains with minimal computational overhead.

<div align='center'>
<img src="../assets/lora.png" class="interpolation-image" alt="LoRA fine-tuning example." width="100%" />
<br>
<em>LoRA fine-tuning examples of text-to-image generation and image-to-image editing under Ghibli-style/Yarn-art-style/Snoopy-style/Irasutoya-style LoRA fine-tuning.</em>
</div>

## 📁 Repository Structure

The necessary scripts for LoRA customization are located within this directory:

| Script | Path | Description |
| :--- | :--- | :--- |
| **Generation** | `train_gen_lora.py` | Fine-tune generation capabilities (e.g., style transfer, character injection). Conditional latents are explicitly set to `0-tensor`. |
| **Editing** | `train_edit_lora.py` | Fine-tune image-to-image editing (e.g., specific object replacement). Requires condition image latents and raw `PIL.Image` for `encode_prompt`. |
| **Inference** | `infer_lora.py` | Script for generating or editing images utilizing the trained LoRA weights via `peft`. |

## 🚀 Training

### 1. Text-to-Image Generation LoRA

For standard generation LoRA (e.g., [Yarn-art-style](https://huggingface.co/datasets/Norod78/Yarn-art-style)), DreamLite acts as a standard diffusion model. The condition image latent `cond_img_in` is replaced with zeros.

```bash
python lora/train_gen_lora.py \
    --model_id "ByteVisionLab/DreamLite-base" \
    --output_dir "./output_lora/yarn" \
    --max_train_steps 2500 \
    --learning_rate 5e-5
```

### 2. Image Editing LoRA

For image editing LoRA (e.g., [Snoopy-style](https://huggingface.co/datasets/showlab/OmniConsistency/viewer/default/Snoopy)), DreamLite utilizes in-context spatial concatenation. This means the model requires both the noisy target latents and the encoded source condition latents.

```bash
python lora/train_edit_lora.py \
    --model_id "ByteVisionLab/DreamLite-base" \
    --output_dir "./output_lora/edit_Snoopy" \
    --max_train_steps 3500 \
    --default_prompt "transfer the image into Snoopy style"
```

### 3. Dataset Customization:
Update the `TODO` block in `train_edit_lora.py` and `train_gen_lora.py`. Your dataloader must yield:
- `target_imgs`: Tensor of ground truth images [B, 3, 1024, 1024].
- `prompts`: List of editing instructions. 
- `source_imgs`: Tensor of condition input images [B, 3, 1024, 1024]. (required for image editing LoRA)
- `source_imgs_pil`: List of original PIL.Image objects (required for encode_prompt in image editing LoRA).
