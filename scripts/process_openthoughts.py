

import random
from datasets import load_dataset
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformers import AutoTokenizer

ds = load_dataset("open-thoughts/OpenThoughts-114k", "default")['train']

# 加载 Qwen/Qwen2.5-7B-Instruct 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 初始化一个列表来存储每个样本的 token 长度
token_lengths = []

# 遍历数据集中的每个样本

math_code_samples = []

for sample in tqdm.tqdm(ds):
    if sample['conversations'][0]['value'].startswith("Return your final response"):
        math_code_samples.append(sample)
    elif sample['conversations'][0]['value'].startswith("Generate an executable Python function"):
        math_code_samples.append(sample)

print(len(math_code_samples))

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

filename = "math_code_long_cot_samples-10k.json"
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(math_code_samples, f, ensure_ascii=False, indent=4)
        
print(len(chosen_sample))

# # 定义范围
# ranges = [
#     # (0, 4000),  
#     # (0, 8000),  
#     # (0, 12000),
#     # (0, 16000),
# ]

# sampled_datasets = {}
# for range_start, range_end in ranges:
#     # 筛选出符合当前范围的样本
#     filtered_samples = [sample for sample in math_code_samples if range_start <= sample['token_length'] <= range_end]
    
#     # 随机采样 10,000 个样本（如果样本不足 10,000，则取全部）
#     if len(filtered_samples) > 10000:
#         sampled_samples = random.sample(filtered_samples, 10_000)
#     else:
#         sampled_samples = filtered_samples
    
#     # 保存采样结果
#     sampled_datasets[f"{range_start}-{range_end}"] = sampled_samples

# for range_name, samples in sampled_datasets.items():
#     filename = f"samples_{range_name}.json"
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(samples, f, ensure_ascii=False, indent=4)
#     print(f"范围 {range_name} 采样了 {len(samples)} 个样本")
    
# # 将 token 长度转换为 numpy 数组
# token_lengths = np.array(token_lengths)

# # 统计 token 长度的分布
# plt.hist(token_lengths, bins=50, edgecolor='black')
# plt.title('Token Length Distribution')
# plt.xlabel('Token Length')
# plt.ylabel('Frequency')
# plt.savefig('tmp.png')

# # 打印一些统计信息
# print(f"Mean token length: {np.mean(token_lengths)}")
# print(f"Median token length: {np.median(token_lengths)}")
# print(f"Max token length: {np.max(token_lengths)}")
# print(f"Min token length: {np.min(token_lengths)}")
