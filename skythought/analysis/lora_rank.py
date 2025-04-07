import os
import re
import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict
import safetensors.torch as sf


# 加载原始模型
original_model_path = "Qwen/Qwen2.5-7B-Instruct"
original_model = AutoModelForCausalLM.from_pretrained(original_model_path)

# 加载全量微调模型
# length, rank = '2k', 64
# length, rank = '2k', 128
# length, rank = '16k', 64
# length, rank = '16k', 128

original_model_path = "Qwen/Qwen2.5-7B-Instruct"
model_path1 = f"skythought/saves/Open-Thoughts-2k-10k/Qwen2.5-7B-Instruct/full/checkpoint-936"
model_path2 = f"skythought/saves/Open-Thoughts-16k-10k/Qwen2.5-7B-Instruct/full/checkpoint-936"

model1 = AutoModelForCausalLM.from_pretrained(model_path1)
model2 = AutoModelForCausalLM.from_pretrained(model_path2)

# 获取原始模型和全量微调模型的权重
original_state_dict = original_model.state_dict()
full_state_dict = model1.state_dict()
lora_state_dict = model2.state_dict()

# 计算全量微调的权重变化
delta_W1 = {}
for key in full_state_dict.keys():
    if key in original_state_dict:
        delta_W1[key] = full_state_dict[key] - original_state_dict[key]

delta_W2 = {}
for key in lora_state_dict.keys():
    if key in original_state_dict:
        delta_W2[key] = lora_state_dict[key] - original_state_dict[key]

# 存储每一层的差距
gaps = []

def extract_layer_number(key):
    # 使用正则表达式提取层号
    match = re.search(r"layers\.(\d+)", key)
    if match:
        return int(match.group(1))
    return -1  # 如果没有层号，返回 -1


param_type_changes = defaultdict(list)

for layer_name in sorted(delta_W2.keys(), key=extract_layer_number):
    if layer_name:
        # 获取 delta_W1 和 delta_W2
        delta1 = delta_W1[layer_name]
        delta2 = delta_W2[layer_name]
        
        # 计算差距（Frobenius 范数）
        gap = (torch.norm(delta1 - delta2) / delta1.norm()).item()
        gaps.append(gap)
        
        param_type = layer_name.split('.')[-3] + '.' + layer_name.split('.')[-2]
        param_type_changes[param_type].append(gap)
        
        # 打印当前层的差距
        print(f"Layer: {layer_name}, Gap: {gap}")
    else:
        # print(f"Layer {layer_name} not found in delta_W1!")
        pass

# 计算并打印平均差距
average_gap = sum(gaps) / len(gaps)
print(f"\nAverage gap between delta_W1 and delta_W2: {average_gap}")


# 计算每一类的平均变化程度
average_changes = {}
for param_type, changes in param_type_changes.items():
    average_changes[param_type] = sum(changes) / len(changes)

# 打印结果
for param_type, avg_change in average_changes.items():
    print(f"Parameter Type: {param_type}, Average Change: {avg_change:.6f}")
    