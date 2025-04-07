

import random
from datasets import load_dataset
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformers import AutoTokenizer
from pathlib import Path


ds = load_dataset("open-thoughts/OpenThoughts-114k", "default")['train']

# 加载 Qwen/Qwen2.5-7B-Instruct 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 初始化一个列表来存储每个样本的 token 长度
token_lengths = []

# 遍历数据集中的每个样本

math_code_samples = []

n_math, n_code = 0, 0
for sample in tqdm.tqdm(ds):
    if sample['conversations'][0]['value'].startswith("Return your final response"):
        math_code_samples.append(sample)
        n_math += 1
    elif sample['conversations'][0]['value'].startswith("Generate an executable Python function"):
        math_code_samples.append(sample)
        n_code += 1

print(n_math, n_code)

max_length = 16384

random.shuffle(math_code_samples)

chosen_sample = []

for sample in tqdm.tqdm(math_code_samples):
    # 拼接所有对话中的 value 内容
    combined_text = sample['conversations'][-1]['value']
    # 对拼接后的文本进行编码
    encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
    # 获取 token 长度
    token_length = encoded['input_ids'].shape[1]
    # 将 token 长度添加到列表中

    if token_length <= max_length:
        chosen_sample.append(sample)
    
    if len(chosen_sample) == 20_000:
        break

filename = "skythought/train/LLaMA-Factory/data/Open-Thoughts/math_code_long_cot_samples-20k.json"

Path(filename).parent.mkdir(parents=True, exist_ok=True)

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(math_code_samples, f, ensure_ascii=False, indent=4)
        
print(len(chosen_sample))