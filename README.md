# DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing

<div align="center">


<!-- [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-DreamLite-yellow)](https://huggingface.co/ByteVisionLab/DreamLite)&nbsp; -->
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/carlofkl/DreamLite)
[![Paper](https://img.shields.io/badge/Paper-DreamLite-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.28713)&nbsp;
[![Project Page](https://img.shields.io/badge/Project_page-More_visualizations-green?logo=bytedance)](https://carlofkl.github.io/dreamlite/)&nbsp;
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=carlofkl.DreamLite)](https://github.com/ByteVisionLab/DreamLite)

</div>

## 🌿 Overview

We introduce **DreamLite**, a compact and unified on-device diffusion model (**0.39B**) that seamlessly supports both **text-to-image generation** and **text-guided image editing** within a single network architecture. 

Built upon a pruned mobile U-Net backbone, DreamLite unifies multimodal conditioning through **In-Context Spatial Concatenation** directly in the latent space. By leveraging progressive step distillation, DreamLite achieves ultra-fast **4-step inference**, capable of generating or editing a **1024×1024** image in ~**3 seconds** on an iPhone 17 Pro (powered by 4-bit Qwen-VL and fp16 VAE+UNet) — operating **fully on-device with zero cloud dependency**.

<div align='center'>
<img src="./assets/cover.png" class="interpolation-image" alt="DreamLite Teaser" width="95%" />
</div>

<br>

<div align='center'>
<img src="./assets/pipeline.png" class="interpolation-image" alt="DreamLite Architecture" width="95%" />
<br>
<em>Figure 1. The overall unified architecture of DreamLite.</em>
</div>

---

## 📰 News

- **[2026.04]** 🎉🎉🎉 We officially released the inference code.
<!-- and model [weights](https://huggingface.co/ByteVisionLab/DreamLite) on Hugging Face. -->
- **[2026.03]** 🎉🎉🎉 DreamLite is publicly announced! Check out our [project page](https://carlofkl.github.io/dreamlite/) and [arXiv paper](https://arxiv.org/abs/2603.28713).

---

## 🎬 On-Device Demo

Experience real-time generation and editing on an iPhone 17 Pro. No internet connection or cloud processing required.

<table align="center">
  <tr>
    <th align="center">Human Portrait & Style Transfer</th>
    <th align="center">Nature Landscape & Background Swap</th>
    <th align="center">Product & Object Replacement</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/demo1.gif" width="280" />
    </td>
    <td align="center">
      <img src="assets/demo2.gif" width="280" />
    </td>
    <td align="center">
      <img src="assets/demo3.gif" width="280" />
    </td>
  </tr>
</table>

> **Note**: If videos fail to render natively on GitHub, please visit our [Project Page](https://carlofkl.github.io/dreamlite/) to watch the full demonstrations.

---

## ⚙️ Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ByteVisionLab/DreamLite.git
cd DreamLite

# Create and activate a conda environment
conda create -n dreamlite python=3.10 -y
conda activate dreamlite

# Install dependencies
pip install -r requirements.txt
```

Ensure the model weights (DreamLite-base and DreamLite-mobile) are placed in the following directory structure:
```
DreamLite/
├── models/
│   ├── DreamLite-base/
│   └── DreamLite-mobile/
```

### 2. Inference via CLI
You can readily generate or edit images utilizing our provided command-line interfaces.
```bash
# ==========================================
# DreamLite-base: 28 Steps (High Fidelity)
# ==========================================
# Text-to-Image Generation
python infer.py --prompt "A close-up of a fire spitting dragon cinematic shot."

# Text-guided Image Editing
python infer.py --prompt "Transfer this image to oil-painting style." --image_path ./inputs/source.png

# ==========================================
# DreamLite-mobile: 4 Steps (Ultra Fast)
# ==========================================
# Text-to-Image Generation
python infer_mobile.py --prompt "A portrait of a young woman with flowers." 

# Text-guided Image Editing
python infer_mobile.py --prompt "Change the background to a dense forest." --image_path ./inputs/source.png
```

### 3. Benchmark Evaluation
We provide comprehensive benchmark evaluation scripts (GenEval & ImgEdit) to facilitate performance comparisons between DreamLite and other state-of-the-art models. Please configure your local dataset paths within `tools/benchmark/infer_geneval.py` and `tools/benchmark/infer_imgedit.py` prior to execution.
```bash
# Run the benchmark evaluation
python tools/benchmark/infer_geneval.py
python tools/benchmark/infer_imgedit.py
```

<!-- ### 4. Inference via Python API (diffusers)
DreamLite is designed to be compatible with standard diffusers pipelines.
```python
import torch
from dreamlite import DreamLiteMobilePipeline
from diffusers.utils import load_image

# Load the distilled model
pipe = DreamLiteMobilePipeline.from_pretrained("ByteVisionLab/DreamLite-mobile", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Task 1: Generation
gen_img = pipe("A serene lake surrounded by mountains at sunset", num_inference_steps=4).images[0]
gen_img.save("output_gen.png")

# Task 2: Editing
source = load_image("inputs/source.png")
edit_img = pipe(prompt="Make the sky more dramatic with orange clouds", image=source, num_inference_steps=4).images[0]
edit_img.save("output_edit.png")
``` -->

### 4. Interactive Gradio Demo

We provide a user-friendly web interface powered by Gradio. You can try our live demo on Hugging Face Spaces, or deploy it locally on your own machine (GPU/CPU).

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/carlofkl/DreamLite)

To run the interactive demo locally:
```bash
# Launch the local web server
python tools/app.py
```



## 🤗 Checkpoints
We offer two distinct variants of the DreamLite model to provide an optimal balance between visual fidelity and on-device inference latency.

> [!NOTE]
> Model weights are currently undergoing safety review. To request early access, please contact us at 📧 klfeng1206@outlook.com


<table>
<tr>
<th align="left">Model Variant</th>
<th align="center">Params</th>
<th align="center">Resolution</th>
<th align="center">Steps</th>
<th align="center">Guidance</th>
<!-- <th align="center">Hugging Face Hub</th> -->
</tr>
<tr>
<td><strong>DreamLite (Base)</strong></td>
<td align="center">0.39B</td>
<td align="center">1024×1024</td>
<td align="center">28</td>
<td align="left">CFG & IMG_CFG</td>
<!-- <td align="center"><a href="https://huggingface.co/ByteVisionLab/DreamLite-base">🤗 Download</a></td> -->
</tr>
<tr>
<td><strong>DreamLite (Mobile)</strong></td>
<td align="center">0.39B</td>
<td align="center">1024×1024</td>
<td align="center">4</td>
<td align="left">No CFG</td>
<!-- <td align="center"><a href="https://huggingface.co/ByteVisionLab/DreamLite-mobile">🤗 Download</a></td> -->
</tr>
</table>

## 📊 Main Results

Quantitative comparison with state-of-the-art methods on generation and editing benchmarks.

<div align='center'>
<img src="./assets/gen.png" class="interpolation-image" alt="generation comparison" width="95%" />
<br>
<em>Text-to-Image generation comparison.</em>
</div>

<br>

<div align='center'>
<img src="./assets/edit.png" class="interpolation-image" alt="editing comparison" width="95%" />
<br>
<em>Text-guided image editing comparison.</em>
</div>

<br>

<table>
  <tr>
    <th>Method</th>
    <th>Params</th>
    <th>GenEval ↑</th>
    <th>DPG ↑</th>
    <th>ImgEdit ↑</th>
    <th>GEdit-EN-Q ↑</th>
  </tr>
  <tr>
    <td>FLUX.1-Dev / Kontext</td>
    <td align="center">12B</td>
    <td align="center">0.67</td>
    <td align="center">84.0</td>
    <td align="center">3.76</td>
    <td align="center">6.79</td>
  </tr>
  <tr>
    <td>BAGEL</td>
    <td align="center">7B</td>
    <td align="center">0.82</td>
    <td align="center">85.1</td>
    <td align="center">3.42</td>
    <td align="center">7.20</td>
  </tr>
  <tr>
    <td>OmniGen2</td>
    <td align="center">4B</td>
    <td align="center">0.80</td>
    <td align="center">83.6</td>
    <td align="center">3.44</td>
    <td align="center">6.79</td>
  </tr>
  <tr>
    <td>LongCat-Image / Edit</td>
    <td align="center">6B</td>
    <td align="center">0.87</td>
    <td align="center">86.6</td>
    <td align="center">4.49</td>
    <td align="center">7.55</td>
  </tr>
  <tr>
    <td>DeepGen1.0</td>
    <td align="center">2B</td>
    <td align="center">0.83</td>
    <td align="center">84.6</td>
    <td align="center">4.03</td>
    <td align="center">7.54</td>
  </tr>
  <tr>
    <td>SANA-1.6B</td>
    <td align="center">1.6B</td>
    <td align="center">0.67</td>
    <td align="center">84.8</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>SANA-0.6B</td>
    <td align="center">0.6B</td>
    <td align="center">0.64</td>
    <td align="center">83.6</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>SnapGen++ (small)</td>
    <td align="center">0.4B</td>
    <td align="center">0.66</td>
    <td align="center">85.2</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>VIBE</td>
    <td align="center">1.6B</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">3.85</td>
    <td align="center">7.28</td>
  </tr>
  <tr>
    <td>EditMGT</td>
    <td align="center">0.96B</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">2.89</td>
    <td align="center">6.33</td>
  </tr>
  <tr>
    <td><strong>DreamLite (Ours)</strong></td>
    <td align="center"><strong>0.39B</strong></td>
    <td align="center"><strong>0.72</strong></td>
    <td align="center"><strong>85.8</strong></td>
    <td align="center"><strong>4.11</strong></td>
    <td align="center"><strong>6.88</strong></td>
  </tr>
</table>

## 🎛️ LoRA Fine-tuning
We provide comprehensive support for LoRA fine-tuning and inference, enabling lightweight customization of DreamLite on your own domain-specific datasets.

For detailed instructions, training scripts, and examples, please refer to our dedicated **[LoRA Fine-Tuning Guide](lora/README.md)**.

## 📑 Open-Source Plan
- [X] Release paper on arXiv
- [X] Release inference code
- [X] Release LoRA training
- [ ] Release model weights on HuggingFace
- [X] Release online demo
- [ ] On-device Deployment Reference


## 🙏 Acknowledgement
We thank the great work from [SDXL](https://github.com/Stability-AI/generative-models), [SnapGen](https://snap-research.github.io/snapgen/), [Qwen](https://qwen.ai/home) and [TAESDXL](https://github.com/madebyollin/taesd). The work is under supervision from Prof. Wangmeng Zuo.


## 🪪 License
**Code**: Apache-2.0

**Model weights**: see WEIGHTS_LICENSE, CC BY-NC 4.0

## 📄 Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:

```bibtex
@article{feng2026dreamlite,
  title={DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing},
  author={Kailai Feng and Yuxiang Wei and Bo Chen and Yang Pan and Hu Ye and Songwei Liu and Chenqian Yan and Yuan Gao},
  journal={arXiv preprint arXiv:2603.28713},
  year={2026}
}
```