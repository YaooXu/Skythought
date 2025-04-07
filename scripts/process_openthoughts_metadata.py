

import random
from datasets import load_dataset
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformers import AutoTokenizer

import os

os.environ['HF_DATASETS_OFFLINE'] = '1'

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


system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:"

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
num_samples = 20_000

for sample in tqdm.tqdm(sub_samples):
    problem = sample['problem']
    
    if sample['domain'] == 'math':
        prompt = math_instruction + problem
        
    # code没有short cot
    # elif sample['domain'] == 'code':
    #     prompt = format_code_prompt(sample)
        
    # long_cot_samples
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
    
    encoded = tokenizer(conversations[-1]['value'], return_tensors='pt', truncation=True, padding=True)
    # 获取 token 长度
    token_length = encoded['input_ids'].shape[1]
    if token_length > max_length:
        continue
        
    # short_cot_samples
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
    
    long_cot_samples.append(long_cot_sample)
    short_cot_samples.append(short_cot_sample)
    
    if len(long_cot_samples) >= num_samples:
        break

long_cot_filename = "math_long_cot_samples-20k.json"
short_cot_filename = "math_short_cot_samples-20k.json"

with open(long_cot_filename, "w") as f:
    json.dump(long_cot_samples, f, indent=4)
    
with open(short_cot_filename, "w") as f:
    json.dump(short_cot_samples, f, indent=4)
    

def analysis_token_length(file_name):
    # 初始化一个列表来存储每个样本的 token 长度
    token_lengths = []

    with open(file_name, 'r') as f:
        samples = json.load(f)
        
    for sample in tqdm.tqdm(samples):
        # 拼接所有对话中的 value 内容
        combined_text = sample['conversations'][-1]['value']
        # 对拼接后的文本进行编码
        encoded = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
        # 获取 token 长度
        token_length = encoded['input_ids'].shape[1]
        # 将 token 长度添加到列表中
        token_lengths.append(token_length)
        
    # 将 token 长度转换为 numpy 数组
    token_lengths = np.array(token_lengths)
    
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.title('Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    # plt.savefig(f'{file_name.split}.png')
    plt.show()

    return token_lengths

# analysis_token_length(long_cot_filename)
# analysis_token_length(short_cot_filename)