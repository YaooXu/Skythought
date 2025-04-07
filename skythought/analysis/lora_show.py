
from collections import defaultdict
from safetensors import safe_open
import torch
from traitlets import default
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float32,
    device_map="cpu"
)
ori_params = dict(model.named_parameters())

lora_path = 'skythought/saves/debug/checkpoint-20/adapter_model.safetensors'

lora_weights = {}
with safe_open(lora_path, framework='pt', device="cpu") as f:
    for key in f.keys():
        lora_weights[key.replace('base_model.model.', '')] = f.get_tensor(key).float()

name_to_changes = defaultdict(list)
for param_name in ori_params:
    if 'bias' in param_name:
        continue
    
    parts = param_name.split('.')
    if len(parts) <= 3:
        continue
    
    name = f'{parts[3]}.{parts[4]}'
    layer = int(parts[2])
    
    lora_a_param_name = param_name.replace('.weight', '.lora_A.weight')
    lora_b_param_name = param_name.replace('.weight', '.lora_B.weight')
    if lora_a_param_name in lora_weights and lora_b_param_name in lora_weights:
        lora_a = lora_weights[lora_a_param_name]
        lora_b = lora_weights[lora_b_param_name]
        
        delta = 2 * lora_b @ lora_a
        
        print(f'Layer {layer} - {name}: {torch.norm(delta) / torch.norm(ori_params[param_name])}')
        
        name_to_changes[name].append(torch.norm(delta) / torch.norm(ori_params[param_name]))

for name, changes in name_to_changes.items():
    print(f'{name}: {sum(changes) / len(changes)}')