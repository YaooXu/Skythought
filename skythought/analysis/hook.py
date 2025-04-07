from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import torch
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

def plot_heatmap(activations, title, save_path=None):
    """
    Plot a heatmap of activation norms across layers and tokens.
    
    Args:
        activations: List of activation norms (n_layers, n_tokens)
        title: Title for the plot
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(16, 8))
    sns.heatmap(activations, cmap="viridis", 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Activation Norm'})
    plt.title(title)
    plt.xlabel("Token Position")
    plt.ylabel("Layer Depth")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


data_path = 'skythought/train/LLaMA-Factory/data/Open-Thoughts/test.json'
with open(data_path, 'r') as f:
    samples = json.load(f)

conversations = []
for sample in samples:
    conversation = [{'role': 'system', 'content': sample['system']}]

    conversation.append({'role': sample['conversations'][0]['from'], 'content': sample['conversations'][0]['value']})

    conversations.append(conversation)
    
# 直接在代码中设置参数，而不是通过命令行
class Args:
    device = "0"  # 默认使用 GPU 0
    # model = "skythought/saves/Open-Thoughts-40k/Qwen2.5-7B-Instruct/lora-64/complete_ckpt"  # 默认模型路径
    model = "skythought/saves/Open-Thoughts-40k/Qwen2.5-7B-Instruct/full"  # 默认模型路径
    max_length = 2048  # 默认最大长度

args = Args()

# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--device", type=str, default="0")
# parser.add_argument(
#     "-m", "--model", type=str, default="skythought/saves/Open-Thoughts-40k/Qwen2.5-7B-Instruct/lora-64/complete_ckpt"
# )
# parser.add_argument("-l", "--max_length", type=int, default=2048)
# args = parser.parse_args()

# full skythought/saves/Open-Thoughts-40k/Qwen2.5-7B-Instruct/full
# lora skythought/saves/Open-Thoughts-40k/Qwen2.5-7B-Instruct/lora-64/complete_ckpt

os.environ["CUDA_VISIBLE_DEVICES"] = args.device


class MLPLayerHook:
    """
    Hook for recording the output of an MLP layer in a LlamaForCausalLM model.
    """

    def __init__(self, layer):
        self.layer = layer
        # self.outputs = []
        self.actition = []

    def __call__(self, module, input, output):
        # print(input[0].shape)
        # self.actition.append(module.act_fn(module.gate_proj(input[0])).cpu())
        self.actition.append(output[0].cpu())

class MLPShiftLayerHook:
    """
    Hook for recording the output of an MLP layer in a LlamaForCausalLM model.
    """

    def __init__(self, layer):
        self.layer = layer
        # self.outputs = []
        self.actition = []

    def __call__(self, module, input, output):
        hidden_state = input[0]
        
        hidden_state_shift = F.pad(hidden_state[:, :-1, :], (0, 0, 1, 0))
        
        if os.environ['SHIFT_VERSION'] == 'v3.5':
            alpha = F.sigmoid(module.scale(torch.concat([hidden_state_shift, hidden_state], dim=-1)))
            shift_gate = module.W(module.R(torch.concat([hidden_state_shift, hidden_state], dim=-1))) * alpha

        if os.environ['SHIFT_VERSION'] == 'v3.5-abl1':
            alpha = F.sigmoid(module.scale(hidden_state))
            shift_gate = module.W(module.R(hidden_state)) * alpha

        if os.environ['SHIFT_VERSION'] == 'v3.5-abl2':
            alpha = F.sigmoid(module.scale(hidden_state_shift))
            shift_gate = module.W(module.R(hidden_state_shift)) * alpha
                           
        ori_gate = module.gate_proj(hidden_state)
        
        final_gate = ori_gate + shift_gate
        
        self.actition.append(module.act_fn(shift_gate).cpu())
        
        # self.actition.append(F.cosine_similarity(ori_gate, shift_gate, dim=1).mean().cpu().item())


# class MLPLayerbackHook:
#     """
#     Hook for recording gradients of MLP layers in LlamaForCausalLM model.
#     新增梯度存储属性
#     """
#     def __init__(self, layer):
#         self.layer = layer
#         self.activations = []
#         self.gradients = []  # 新增梯度存储列表

#     def __call__(self, module, grad_input, grad_output):  # 修改为梯度回调函数
#         # 捕获梯度输出（grad_output对应前向传播输出的梯度）
#         if grad_output[0] is not None:
#             self.gradients.append(grad_output[0].detach().cpu())


def register_mlp_hooks(model, Hook=MLPLayerHook):
    """
    Registers MLPLayerHook instances to all MLP layers in the given LlamaForCausalLM model.

    Args:
        model: The LlamaForCausalLM model.

    Returns:
        A list of MLPLayerHook instances.
    """
    hooks = []
    for name, module in model.named_modules():
        if name.endswith("mlp"):
            hook = Hook(module)
            # 注册前向钩子来捕获输出
            module.register_forward_hook(hook)
            # 注册反向钩子来捕获梯度
            # module.register_full_backward_hook(hook)
            hooks.append(hook)
    return hooks
    
    
output_folder = os.path.join(args.model, "hook_output")

if 'shift' in args.model:
    print('using shift!')
    # os.environ['SHIFT_VERSION'] = 'v3.5'
    # os.environ['SHIFT_VERSION'] = 'v3.5-abl1'
    os.environ['SHIFT_VERSION'] = 'v3.5-abl2'
    from llamafactory.custom_models.modeling_qwen2_shift_gate import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(
        args.model, device_map='auto', torch_dtype='auto', attn_implementation="flash_attention_2"
    )
    mlp_hooks = register_mlp_hooks(model, MLPShiftLayerHook)
    
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map='auto', torch_dtype='auto', attn_implementation="flash_attention_2"
    )
    mlp_hooks = register_mlp_hooks(model, MLPLayerHook)
    
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.eval()

for i, conversation in enumerate(conversations):
    text = tokenizer.apply_chat_template(conversation, tokenize=False)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**model_inputs,
                                max_new_tokens =2048,
                                do_sample=False)
    
    activation_by_layer = []
    layer_diff_act_norm = []
    for k, hook in enumerate(mlp_hooks):
        act = hook.actition[-1][0].float().cpu().numpy()
        diff_act = np.diff(act, axis=0)
        diff_act_norm = np.linalg.norm(diff_act, axis=-1) / np.linalg.norm(act[1:], axis=-1)
        layer_diff_act_norm.append(diff_act_norm.mean())
        
    print(layer_diff_act_norm)
    print(np.array(layer_diff_act_norm).mean())
    plt.plot(layer_diff_act_norm)
    break
