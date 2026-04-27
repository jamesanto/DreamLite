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

import gradio as gr
import torch
from PIL import Image
from diffusers.utils import load_image

# 导入你的两个 Pipeline
from dreamlite import DreamLitePipeline 
from dreamlite import DreamLiteMobilePipeline 

# ==========================================
# 1. 全局配置与模型缓存管理
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 定义可用的模型及其对应的 Pipeline 类
MODEL_CONFIGS = {
    "DreamLite-base": {
        "path": "models/DreamLite-base",
        "pipeline_class": DreamLitePipeline
    },
    "DreamLite-mobile": {
        "path": "models/DreamLite-mobile",
        "pipeline_class": DreamLiteMobilePipeline
    }
}

# 用于在内存中缓存已加载的模型，避免重复加载
loaded_models = {}
current_model_name = None

def load_or_get_model(model_name):
    """
    负责按需加载模型：
    1. 如果内存里已经有这个模型，直接返回。
    2. 如果没有，则加载它，并且为了防止显存爆炸，清理掉之前加载的其他模型（如果显存足够可以不清理，这里采取保守清理策略）。
    """
    global current_model_name
    
    # 命中缓存，直接返回
    if model_name in loaded_models:
        return loaded_models[model_name]
        
    # 如果要加载新模型，先卸载旧模型释放显存
    if current_model_name and current_model_name != model_name:
        print(f"Unloading {current_model_name} to free memory...")
        del loaded_models[current_model_name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"Loading {model_name}...")
    config = MODEL_CONFIGS[model_name]
    PipelineClass = config["pipeline_class"]
    model_path = config["path"]
    
    try:
        pipe = PipelineClass.from_pretrained(model_path, torch_dtype=dtype)
        pipe = pipe.to(device)
        loaded_models[model_name] = pipe
        current_model_name = model_name
        print(f"{model_name} Loaded Successfully!")
        return pipe
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

# ==========================================
# 2. 预加载默认模型 (可选，加速初次打开页面的体验)
# ==========================================
# 我们默认先加载 base 版本
load_or_get_model("DreamLite-base")

# ==========================================
# 3. 定义推理函数
# ==========================================
def generate_image(
    model_choice, 
    prompt, 
    image, 
    num_inference_steps, 
    guidance_scale, 
    image_guidance_scale, 
    seed
):
    # 动态获取当前选择的模型
    pipe = load_or_get_model(model_choice)
    if pipe is None:
        raise gr.Error(f"Failed to load model: {model_choice}. Check the logs for details.")

    # 强制将种子转为 Tensor Generator 以保证可复现
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # 将 Gradio 传入的图片 (如果有的话) 转换为 PIL 格式
    input_image = image if image is not None else None
    
    # 调用对应的 Pipeline
    out = pipe(
        prompt=prompt,
        image=input_image,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    
    return out

# ==========================================
# 4. 搭建 Gradio 页面
# ==========================================
with gr.Blocks(title="DreamLite Demo") as demo:
    gr.Markdown("# 🌟 DreamLite: Efficient On-Device Generation and Editing")
    gr.Markdown("Select a model version, then generate images from text or upload an image to edit it based on instructions.")
    
    with gr.Row():
        with gr.Column():
            # 新增：模型选择下拉框
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="DreamLite-base", # 默认选中 base
                label="Select Model Version",
                interactive=True
            )
            
            # 输入组件
            prompt_input = gr.Textbox(label="Prompt / Instruction", placeholder="e.g. A photo of a dog...", lines=3)
            image_input = gr.Image(type="pil", label="Input Image (Optional for Editing)")
            
            with gr.Accordion("Advanced Options", open=False):
                steps_slider = gr.Slider(minimum=1, maximum=50, value=30, step=1, label="Inference Steps")
                guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="Guidance Scale")
                img_guidance_slider = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.1, label="Image Guidance Scale")
                seed_slider = gr.Slider(minimum=0, maximum=999999, value=42, step=1, label="Seed")
                
            submit_btn = gr.Button("Generate / Edit", variant="primary")
            
        with gr.Column():
            # 输出组件
            output_image = gr.Image(type="pil", label="Output Image")

    # 绑定点击事件 (注意 inputs 列表增加了 model_dropdown 作为第一个参数)
    submit_btn.click(
        fn=generate_image,
        inputs=[model_dropdown, prompt_input, image_input, steps_slider, guidance_slider, img_guidance_slider, seed_slider],
        outputs=[output_image]
    )
    
    # 示例区 (同步加上对应的模型选择)
    gr.Examples(
        examples=[
            ["DreamLite-base", "A close-up of a fire spitting dragon, cinematic shot.", None, 28, 7.5, 1.0, 42],
            ["DreamLite-mobile", "Make it look like a pencil sketch", "assets/example.jpg", 4, 1.0, 1.0, 42] 
        ],
        inputs=[model_dropdown, prompt_input, image_input, steps_slider, guidance_slider, img_guidance_slider, seed_slider]
    )

# ==========================================
# 5. 启动应用
# ==========================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)