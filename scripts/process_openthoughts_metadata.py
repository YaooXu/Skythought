

import random
from datasets import load_dataset
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformers import AutoTokenizer

import os
import re
from pathlib import Path

def set_seed(seed):
    random.seed(seed)         
    np.random.seed(seed)      

set_seed(42)

def format_code_prompt(x):
    formatted_prompt = ""

    data = json.loads(x["test_cases"])
    if not data.get("fn_name"):
        formatted_prompt += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition. "  # noqa
    else:
        formatted_prompt += (
            "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution. "  # noqa
        )

    formatted_prompt += x["problem"]
    if x["starter_code"] is not None:
        data = x["starter_code"]
        data = "\n" + data
        formatted_prompt += data
    return formatted_prompt


system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

math_instruction = "Return your final response within \\boxed{}. "

long_cot_response = "<|begin_of_thought|>\n\n{thought}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n{solution}\n\n<|end_of_solution|>"

ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata")['train']

# 加载 Qwen/Qwen2.5-7B-Instruct 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 初始化一个列表来存储每个样本的 token 长度
token_lengths = []

sub_samples = ds.filter(lambda example: example['domain'] in ['math'])

print(len(sub_samples))

sub_samples = sub_samples.shuffle()

long_cot_samples = []
short_cot_samples = []

max_length = 16384
num_samples = 80_000  # 我们只需要收集80k，从中可以提取60k和40k

n_code, n_math = 0, 0

for sample in tqdm.tqdm(sub_samples):
    problem = sample['problem']
    
    if sample['domain'] == 'math':
        prompt = math_instruction + problem
        
    # 处理long_cot样本
    conversations = [
        {
            "from": "user",
            "value": prompt
        },
        {
            "from": "assistant",
            "value": long_cot_response.format(thought=sample['deepseek_reasoning'], solution=sample['deepseek_solution'])
        }
    ]
    long_cot_sample = {
        "system": system_prompt,
        "conversations": conversations
    }
    
    # 处理short_cot样本
    conversations = [
        {
            "from": "user",
            "value": prompt
        },
        {
            "from": "assistant",
            "value": sample['ground_truth_solution']
        }
    ]
    short_cot_sample = {
        "system": system_prompt,
        "conversations": conversations
    }
    
    # 检查token长度
    combined_text = conversations[0]['value'] + conversations[-1]['value']
    encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
    token_length = encoded['input_ids'].shape[1]
    
    if token_length + 32 > max_length:
        continue
        
    if short_cot_sample['conversations'][1]['value'].startswith('### Solution Code'):
        n_code += 1
    else:
        n_math += 1
        
    long_cot_samples.append(long_cot_sample)
    short_cot_samples.append(short_cot_sample)
    
    if len(long_cot_samples) >= num_samples:
        break

print(f"Code samples: {n_code}, Math samples: {n_math}")

# 保存不同规模的数据
output_dir = "skythought/train/LLaMA-Factory/data/Open-Thoughts/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 保存80k数据（完整数据集）
with open(output_dir + "math_long_cot_samples-80k.json", "w") as f:
    json.dump(long_cot_samples, f, indent=4)
with open(output_dir + "math_short_cot_samples-80k.json", "w") as f:
    json.dump(short_cot_samples, f, indent=4)

# 从80k中提取前60k
with open(output_dir + "math_long_cot_samples-60k.json", "w") as f:
    json.dump(long_cot_samples[:60000], f, indent=4)
with open(output_dir + "math_short_cot_samples-60k.json", "w") as f:
    json.dump(short_cot_samples[:60000], f, indent=4)

# 从80k中提取前40k
with open(output_dir + "math_long_cot_samples-40k.json", "w") as f:
    json.dump(long_cot_samples[:40000], f, indent=4)
with open(output_dir + "math_short_cot_samples-40k.json", "w") as f:
    json.dump(short_cot_samples[:40000], f, indent=4)


with open(output_dir + "math_long_cot_samples-10k.json", "w") as f:
    json.dump(long_cot_samples[:10000], f, indent=4)
with open(output_dir + "math_short_cot_samples-10k.json", "w") as f:
    json.dump(short_cot_samples[:10000], f, indent=4)


# def analysis_token_length(file_name):
#     # 初始化一个列表来存储每个样本的 token 长度
#     token_lengths = []

#     with open(file_name, 'r') as f:
#         samples = json.load(f)
        
#     for sample in tqdm.tqdm(samples):
#         # 拼接所有对话中的 value 内容
#         combined_text = sample['conversations'][-1]['value']
#         # 对拼接后的文本进行编码
#         encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
#         # 获取 token 长度
#         token_length = encoded['input_ids'].shape[1]
#         # 将 token 长度添加到列表中
#         token_lengths.append(token_length)
        
#     # 将 token 长度转换为 numpy 数组
#     token_lengths = np.array(token_lengths)
    
#     plt.hist(token_lengths, bins=50, edgecolor='black')
#     plt.title('Token Length Distribution')
#     plt.xlabel('Token Length')
#     plt.ylabel('Frequency')
#     # plt.savefig(f'{file_name.split}.png')
#     plt.show()

#     return token_lengths

# analysis_token_length(long_cot_filename)
# analysis_token_length(short_cot_filename)