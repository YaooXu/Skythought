

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


with open(output_dir + "math_long_cot_samples-20k.json", "w") as f:
    json.dump(long_cot_samples[:20000], f, indent=4)
with open(output_dir + "math_short_cot_samples-20k.json", "w") as f:
    json.dump(short_cot_samples[:20000], f, indent=4)


with open(output_dir + "math_long_cot_samples-10k.json", "w") as f:
    json.dump(long_cot_samples[:10000], f, indent=4)
with open(output_dir + "math_short_cot_samples-10k.json", "w") as f:
    json.dump(short_cot_samples[:10000], f, indent=4)


long_cot_filename = output_dir + "math_long_cot_samples-10k.json"
short_cot_filename = output_dir + "math_short_cot_samples-10k.json"
def analysis_token_length(file_path1, file_path2, label1, label2):
    """
    Analyzes token lengths from two JSON files and plots their distributions
    on a single histogram with shared x and y axes.

    Args:
        file_path1 (str): Path to the first JSON file.
        file_path2 (str): Path to the second JSON file.
        label1 (str): Label for the first dataset in the plot legend.
        label2 (str): Label for the second dataset in the plot legend.
    """
    plt.figure(figsize=(5, 6))

    all_token_lengths = [] # To store all lengths for global x-axis range

    # --- Process the first file ---
    token_lengths1 = []
    try:
        with open(file_path1, 'r') as f:
            samples1 = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path1}. Skipping this dataset.")
        samples1 = [] # Set to empty list to avoid errors
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path1}. Skipping this dataset.")
        samples1 = [] # Set to empty list to avoid errors

    for sample in samples1:
        combined_text = ''
        # Assumes conversations is a list and value is in the last element
        if 'conversations' in sample and isinstance(sample['conversations'], list) and len(sample['conversations']) > 0:
            if 'value' in sample['conversations'][-1]:
                combined_text = sample['conversations'][-1]['value']
        # Fallback for simpler JSON structures where 'value' might be directly in the sample
        elif 'value' in sample:
            combined_text = sample['value']

        if combined_text:
            encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
            length = encoded['input_ids'].shape[1]
            token_lengths1.append(length)
            all_token_lengths.append(length)

    # --- Process the second file ---
    token_lengths2 = []
    try:
        with open(file_path2, 'r') as f:
            samples2 = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path2}. Skipping this dataset.")
        samples2 = []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path2}. Skipping this dataset.")
        samples2 = []

    for sample in samples2:
        combined_text = ''
        if 'conversations' in sample and isinstance(sample['conversations'], list) and len(sample['conversations']) > 0:
            if 'value' in sample['conversations'][-1]:
                combined_text = sample['conversations'][-1]['value']
        elif 'value' in sample:
            combined_text = sample['value']

        if combined_text:
            encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
            length = encoded['input_ids'].shape[1]
            token_lengths2.append(length)
            all_token_lengths.append(length)

    # --- Determine common bins for consistent x-axis ---
    # if all_token_lengths:
    #     max_len = np.max(all_token_lengths)
    #     min_len = np.min(all_token_lengths)
    #     bins = np.linspace(min_len, max_len, 50) # Use 50 bins across the whole range
    # else:
    #     bins = 50 # Default if no data to avoid errors
    bins = np.linspace(0, 16000, 50)
    # --- Plot both histograms on the same figure ---
    # `bins` ensures both histograms use the same x-axis binning.
    # `edgecolor='black'` adds a black border to each bar.
    # `alpha=0.7` makes the bars semi-transparent so overlaps are visible.
    # `label` is used for the legend.
    # The return values `counts` and `_` (bin edges are not directly used here)
    counts1, _, _ = plt.hist(token_lengths1, bins=bins, edgecolor='black', alpha=0.7, label=label1)
    counts2, _, _ = plt.hist(token_lengths2, bins=bins, edgecolor='black', alpha=0.7, label=label2)

    # --- Set common Y-axis limits ---
    max_freq = 0
    if len(counts1) > 0:
        max_freq = max(max_freq, np.max(counts1))
    if len(counts2) > 0:
        max_freq = max(max_freq, np.max(counts2))

    if max_freq > 0:
        plt.ylim(0, max_freq * 1.05) # Add 5% buffer on top for visual clarity

    plt.title('Token Length Distribution Comparison')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.legend() # Displays the labels from the 'label' argument in plt.hist
    plt.grid(axis='y', alpha=0.75) # Adds a light horizontal grid for readability
    plt.show() # Displays the plot

# analysis_token_length(long_cot_filename, short_cot_filename, 'Long COT', 'Short COT')