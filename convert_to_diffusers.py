import torch
from modules.model_utils import load_model

def convert_to_diffusers(output_dir="./models/DreamLite-mobile"):
    print("Loading original weights...")
    # 这里复用你现有的加载逻辑，确保加载出完整的 pipeline
    pipeline = load_model(
        model_path="models/20-40000", # 替换为你想导出的实际 pt 文件
        device="cpu", # 导出时用 CPU 即可
        dtype=torch.bfloat16,
        mode="mobile" # 或者 mobile
    )
    
    print(f"Saving standard diffusers model to {output_dir}...")
    # 调用 diffusers 的原生 save_pretrained 方法
    # 这会在 output_dir 下创建 text_encoder, vae, unet, scheduler 等子文件夹
    # 并在根目录生成 model_index.json
    pipeline.save_pretrained(output_dir)
    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_diffusers()