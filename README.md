# DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing

## 🌿 Overview

We propose **DreamLite**, a compact unified on-device diffusion model (**0.39B**) that supports both **text-to-image generation** and **text-guided image editing** within a single network. DreamLite is built on a pruned mobile U-Net backbone and unifies conditioning through **In-Context spatial concatenation** in the latent space. By employing step distillation, DreamLite achieves **4-step inference**, generating or editing a **1024×1024** image in ~**3 seconds (using 4-bit Qwen VL and fp16 VAE+UNet)** on an iPhone 17 Pro — fully on-device, no cloud required.


## ⚙️ Getting Started

### Requirements

```bash
# Clone the repository
git clone https://github.com/ByteVisionLab/DreamLite.git
cd DreamLite

# Create conda environment
conda create -n dreamlite python=3.10 -y
conda activate dreamlite

# Install dependencies
pip install -r requirements.txt
```

### Inference
```
```

<!-- ```python
from dreamlite import DreamLitePipeline

# Text-to-Image Generation
pipe = DreamLitePipeline.from_pretrained("DreamLite/dreamlite-v1")
image = pipe("A serene lake surrounded by mountains at sunset", num_inference_steps=4)
image.save("output.png")

# Text-guided Image Editing
from PIL import Image
source = Image.open("input.png")
edited = pipe.edit(source, "Make the sky more dramatic with orange clouds", num_inference_steps=4)
edited.save("edited.png")
``` -->


## 🤗 Checkpoints

<table>
  <tr>
    <th style="width: 200px;">Model</th>
    <th>Params</th>
    <th>Resolution</th>
    <th>Steps</th>
    <th>HuggingFace</th>
  </tr>
  <tr>
    <td align="center">DreamLite (Base)</td>
    <td align="center">0.39B</td>
    <td align="center">1024×1024</td>
    <td align="center">28</td>
    <td align="center"><a href="https://huggingface.co/DreamLite">🤗 To be released</a></td>
  </tr>
  <tr>
    <td align="center">DreamLite (Distilled)</td>
    <td align="center">0.39B</td>
    <td align="center">1024×1024</td>
    <td align="center">4</td>
    <td align="center"><a href="https://huggingface.co/DreamLite">🤗 To be released</a></td>
  </tr>
</table>


## 📊 Main Results

Quantitative comparison with state-of-the-art methods on generation and editing benchmarks.

<!-- <div align='center'>
<img src="./assets/gen.png" class="interpolation-image" alt="generation comparison" width="95%" />
<br>
<em>Text-to-Image generation comparison.</em>
</div>

<br>

<div align='center'>
<img src="./assets/edit.png" class="interpolation-image" alt="editing comparison" width="95%" />
<br>
<em>Text-guided image editing comparison.</em>
</div> -->

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


## 📑 Open-Source Plan

- [X] Release paper on arXiv
- [ ] Release inference code
- [ ] Release model weights on HuggingFace
- [ ] Release online demo


## Acknowledgement

We thank the great work from [SDXL](https://github.com/Stability-AI/generative-models), [SnapGen](https://snap-research.github.io/snapgen/), [Qwen](https://qwen.ai/home) and [TAESDXL](https://github.com/madebyollin/taesd). The work is under supervision from Prof. Wangmeng Zuo.


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