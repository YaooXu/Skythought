

import random
from datasets import load_dataset
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import copy
import os
import re

os.environ['HF_HOME'] = '/mnt/workspace/user/sunwangtao/.cache/huggingface'

def replace_thought_content(text):
    return re.sub(
        r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
        "<|begin_of_thought|>\n\n<|end_of_thought|>",
        text,
        flags=re.DOTALL
    )

def set_seed(seed):
    random.seed(seed)         
    np.random.seed(seed)      

set_seed(42)

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
    # elif sample['conversations'][0]['value'].startswith("Generate an executable Python function"):
    #     math_code_samples.append(sample)
    #     n_code += 1

print(n_math, n_code)

max_length = 16384

random.shuffle(math_code_samples)

long_cot_samples = []
short_cot_samples = []

for sample in tqdm.tqdm(math_code_samples):
    # 拼接所有对话中的 value 内容
    combined_text = sample['conversations'][0]['value'] + sample['conversations'][-1]['value']
    # 对拼接后的文本进行编码
    encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
    # 获取 token 长度
    token_length = encoded['input_ids'].shape[1]
    # 将 token 长度添加到列表中

    if token_length + 32 <= max_length:
        long_cot_samples.append(sample)

        short_sample = copy.deepcopy(sample)
        short_sample['conversations'][-1]['value'] = replace_thought_content(sample['conversations'][-1]['value'])

        short_cot_samples.append(short_sample)

    if len(long_cot_samples) == 20_000:
        break

long_filename = "skythought/train/LLaMA-Factory/data/Open-Thoughts/math_long_cot_samples-20k.json"
short_filename = "skythought/train/LLaMA-Factory/data/Open-Thoughts/math_short_cot_samples-20k.json"

Path(long_filename).parent.mkdir(parents=True, exist_ok=True)

with open(long_filename, 'w', encoding='utf-8') as f:
    json.dump(long_cot_samples, f, ensure_ascii=False, indent=4)

with open(short_filename, 'w', encoding='utf-8') as f:
    json.dump(short_cot_samples, f, ensure_ascii=False, indent=4)


# with open(long_filename, 'r') as f:
#     long_cot_samples = json.load(f)

# with open(short_filename, 'r') as f:
#     short_cot_samples = json.load(f)

# # 分别统计 token 数量
# long_lengths = []
# short_lengths = []

# for s in tqdm.tqdm(long_cot_samples):
#     combined_text = s['conversations'][0]['value'] + s['conversations'][-1]['value']
#     encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
#     long_lengths.append(encoded['input_ids'].shape[1])

# for s in tqdm.tqdm(short_cot_samples):
#     combined_text = s['conversations'][0]['value'] + s['conversations'][-1]['value']
#     encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
#     short_lengths.append(encoded['input_ids'].shape[1])

# # 可视化
# plt.figure(figsize=(10, 6))
# plt.hist(long_lengths, bins=50, alpha=0.6, label='Long CoT', color='blue')
# plt.hist(short_lengths, bins=50, alpha=0.6, label='Short CoT', color='orange')
# plt.xlabel("Token Length")
# plt.ylabel("Frequency")
# plt.title("Token Length Distribution: Long CoT vs Short CoT")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # 打印基本统计
# print(f"📊 Long CoT 平均长度: {np.mean(long_lengths):.2f} ± {np.std(long_lengths):.2f}")
# print(f"📊 Short CoT 平均长度: {np.mean(short_lengths):.2f} ± {np.std(short_lengths):.2f}")
