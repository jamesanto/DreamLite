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

import torch
from dreamlite import DreamLitePipeline

pipe = DreamLitePipeline.from_pretrained("models/DreamLite-base", torch_dtype=torch.bfloat16).to("cuda")

# 1. Load LoRA Weights
lora_path = "models/lora/anime_style.safetensors"
pipe.load_lora_weights(lora_path, adapter_name="anime")

# 2. Set LoRA Scale
pipe.set_adapters(["anime"], adapter_weights=[0.8])

# 3. Inference
image = pipe(
    prompt="A girl in the forest", 
    num_inference_steps=28
).images[0]

image.save("lora_output.png")
