# 🎛️ DreamLite LoRA Fine-Tuning Guide

This document provides instructions for training and performing inference with Low-Rank Adaptation (LoRA) on the DreamLite model. LoRA allows you to efficiently fine-tune DreamLite for specific artistic styles, subjects, or domains with minimal computational overhead.

<div align='center'>
<img src="../assets/lora.png" class="interpolation-image" alt="LoRA fine-tuning example. The left image is the original image, and the right images are the fine-tuned images with yarn art style LoRA." width="100%" />
</div>

## 📁 Repository Structure

The necessary scripts for LoRA customization are located within this directory:

| Script | Path | Description |
| :--- | :--- | :--- |
| **Training** | `train_lora.py` | Core script for executing LoRA fine-tuning on custom datasets. |
| **Inference** | `infer_lora.py` | Script for generating or editing images utilizing the trained LoRA weights. |

## 🚀 Training

To initiate LoRA fine-tuning, run the `train_lora.py` script. You can adjust hyperparameters such as learning rate, batch size, and rank (`--rank`) within the script or via command-line arguments (depending on your setup).

```bash
# Example command for launching LoRA training
python lora/train_lora.py \
    --model_id="./models/DreamLite-base" \
    --output_dir="./output/output_lora/yarn" \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --max_train_steps=2500