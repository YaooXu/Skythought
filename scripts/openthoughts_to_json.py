

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

ds = load_dataset("open-thoughts/OpenThoughts-114k", "default")['train']

output_path = 'skythought/train/LLaMA-Factory/data/Open-Thoughts/openthoughts-114k.json'

data_list = []
for item in ds:
    data_list.append(item)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"Dataset saved to JSON file: {output_path}")