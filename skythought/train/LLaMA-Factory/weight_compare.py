from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
import os

hidden_size = None


def load_models(base_model_path, sft_model_path):
    """
    加载两个模型，使用 AutoModelForCausalLM 并强制为 fp32
    """
    model_a = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32, device_map="cpu")
    model_b = AutoModelForCausalLM.from_pretrained(sft_model_path, torch_dtype=torch.float32, device_map="cpu")

    global hidden_size
    hidden_size = model_a.config.hidden_size

    return model_a, model_b


def compute_cosine_similarity(param_a, param_b):
    """
    计算两个参数张量的余弦相似度
    """
    param_a = param_a.detach().cpu()
    param_b = param_b.detach().cpu()

    if param_a.shape[1] != hidden_size:
        param_a = param_a.T
        param_b = param_b.T
        assert param_a.shape[1] == hidden_size  # keep the second dim is 4096

    return F.cosine_similarity(param_a, param_b, dim=1).mean().item()


def plot_embedding_similarity(
    input_embed_changes,
    output_embed_changes,
    input_embed_norm,
    output_embed_norm,
    name="similarity",
    save_path="embedding_similarity.png",
    sim_range=(0.95, 1.0),
):
    """
    绘制 Embedding 的相似度分布柱状图，并保存为文件
    """
    plt.figure(figsize=(8, 6))
    bins = np.linspace(sim_range[0], sim_range[1], 50)  # 相似度范围 [0, 1]

    plt.hist([input_embed_changes, output_embed_changes], bins, alpha=0.5, label=["Input Embedding", "Output Embedding"])

    with open(save_path.replace(".png", ".txt"), "w") as f:
        f.write(f"Input Embedding avg norm: {input_embed_norm:.4f}\n")
        f.write(f"Input Embedding avg {name}: {input_embed_changes.mean():.4f}\n")
        f.write(f"Output Embedding avg norm: {output_embed_norm:.4f}\n")
        f.write(f"Output Embedding avg {name}: {output_embed_changes.mean():.4f}\n")

    plt.title(f"{name} Distribution - Embedding")
    plt.xlabel(f"{name}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Embedding similarity distribution saved to {save_path}")
    plt.close()


def analyze_embedding_similarity(model_a, model_b, name="input_embeddings"):
    """
    计算 Embedding 层的相似度分布，仅针对 Embedding 范数大于 0.1 的 token 进行计算。

    Args:
        model_a: 第一个模型。
        model_b: 第二个模型。
        name: "input_embeddings" 或 "output_embeddings"。

    Returns:
        similarity_vector: 余弦相似度，仅包含满足条件的 token。
        rel_dis: 相对距离变化，仅包含满足条件的 token。
        idx_mapping: 满足范数条件的 token 索引。
    """
    # 获取对应的 Embedding
    if name == "input_embeddings":
        emb_a = model_a.get_input_embeddings().weight.detach().cpu()
        emb_b = model_b.get_input_embeddings().weight.detach().cpu()
    else:
        emb_a = model_a.get_output_embeddings().weight.detach().cpu()
        emb_b = model_b.get_output_embeddings().weight.detach().cpu()

    print(emb_a.shape, emb_b.shape)
    
    # 检查形状
    if emb_a.shape != emb_b.shape:
        raise ValueError("Embedding shapes do not match!")

    # 计算每个 token 的范数
    norms_a = torch.norm(emb_a, p=2, dim=-1)  # [vocab_size]

    # 找出范数大于 0.01 的 token 索引
    valid_indices = (norms_a > 0.01).nonzero(as_tuple=True)[0]  # 获取有效索引

    # 仅选择满足条件的 Embedding
    emb_a_filtered = emb_a[valid_indices]
    emb_b_filtered = emb_b[valid_indices]
    emb_b_avg_norm = emb_b_filtered.norm(dim=-1).mean().item()

    # 逐 token 计算余弦相似度
    similarity = F.cosine_similarity(emb_a_filtered, emb_b_filtered, dim=1).numpy()  # [filtered_vocab_size]

    # 计算相对变化
    rel_dis = 1.0 - torch.norm(emb_a_filtered - emb_b_filtered, p=2, dim=-1) / (torch.norm(emb_a_filtered, p=2, dim=-1) + 1e-8)
    rel_dis = rel_dis.numpy()

    # 将索引映射为 NumPy 数组
    idx_mapping = valid_indices.numpy()

    return similarity, rel_dis, emb_b_avg_norm


def analyze_layerwise_similarity(model_a, model_b, pattern):
    """
    按照 pattern 对每层参数计算相似度，并返回每层的相似度列表
    pattern: 用于匹配 self-attn 或 mlp 的参数，例如 r"layers\.(\d+)\.(self_attn|mlp)\."
    """
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    shared_params = set(params_a.keys()) & set(params_b.keys())
    layerwise_similarities = defaultdict(list)
    layerwise_relative_dis = defaultdict(list)

    for param_name in shared_params:
        if 'bias' in param_name:
            continue
        
        match = re.match(pattern, param_name)
        if match:
            layer_idx = int(match.group(1))  # 提取层号
            sim = compute_cosine_similarity(params_a[param_name], params_b[param_name])
            rel_dis = 1.0 - torch.dist(params_a[param_name], params_b[param_name], p=2) / torch.norm(params_a[param_name], p=2)
            rel_dis = rel_dis.detach().cpu().numpy()
            layerwise_similarities[layer_idx].append(sim)
            layerwise_relative_dis[layer_idx].append(rel_dis)

    # layerwise_similarities = {
    #     layer_idx: sum(sim_list) / len(sim_list) for layer_idx, sim_list in layerwise_similarities.items()
    # }
    # layerwise_relative_dis = {
    #     layer_idx: sum(dis_list) / len(dis_list) for layer_idx, dis_list in layerwise_relative_dis.items()
    # }

    return layerwise_similarities, layerwise_relative_dis


def plot_layerwise_average_change(
    self_attn_changes, mlp_changes, name="similarity", save_path="layerwise_similarity.png", sim_range=(0.95, 1.0)
):
    """
    绘制每层 Self-Attention 和 MLP 平均相似度的折线图
    """
    layers = sorted(set(self_attn_changes.keys()).union(mlp_changes.keys()))

    list_self_attn_changes = np.array([self_attn_changes.get(layer) for layer in layers]).flatten()
    list_mlp_changes = np.array([mlp_changes.get(layer) for layer in layers]).flatten()

    content = f"Self-Attention avg {name}={np.mean(list_self_attn_changes):.4f}, MLP avg {name}={np.mean(list_mlp_changes):.4f}\n"
    for layer in layers:
        line = (
            f"Layer {layer}: Self-Attention {name}={np.mean(self_attn_changes.get(layer)):.4f},"
            f"MLP {name}={np.mean(mlp_changes.get(layer)):.4f}"
        )
        content += line + "\n"
    with open(save_path.replace(".png", ".txt"), "w") as f:
        f.write(content)

    # 绘图
    plt.figure(figsize=(12, 6))

    bins = np.linspace(sim_range[0], sim_range[1], 10)  # 相似度范围 [0, 1]
    
    plt.hist([list_self_attn_changes, list_mlp_changes], bins, alpha=0.5, label=["self-attn", "mlp"])

    # 添加轴标签和标题
    plt.title(f"{name} Distribution - Self-Attn and MLP")
    plt.xlabel(f"{name}")
    plt.ylabel("Frequency")
 
    plt.legend()

    # 保存并展示图像
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Layer-wise {name} bar chart saved to {save_path}")
    plt.show()


def analysis_parameters(base_model_path, sft_model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    print("Loading models...")
    model_a, model_b = load_models(base_model_path, sft_model_path)

    # 2. 分析 Embedding 层
    print("Analyzing Embedding similarity...")
    input_embed_sim, input_embed_dis, input_embed_avg_norm = analyze_embedding_similarity(model_a, model_b, "input_embeddings")
    output_embed_sim, output_embed_dis, output_embed_avg_norm = analyze_embedding_similarity(
        model_a, model_b, "output_embeddings"
    )
    print("Plotting Output Embedding similarity distribution...")

    # 3. 分析 Self-Attention 参数逐层相似度
    print("Analyzing Self-Attention similarity per layer...")
    self_attn_layerwise_sim, self_attn_layerwise_rel_dis = analyze_layerwise_similarity(
        model_a, model_b, r"model\.layers\.(\d+)\.self_attn\."
    )

    # 4. 分析 MLP 参数逐层相似度
    print("Analyzing MLP similarity per layer...")
    mlp_layerwise_sim, mlp_layerwise_rel_dis = analyze_layerwise_similarity(model_a, model_b, r"model\.layers\.(\d+)\.mlp\.")

    sim_range = (0.8, 1.0)
    plot_embedding_similarity(
        input_embed_sim,
        output_embed_sim,
        input_embed_avg_norm,
        output_embed_avg_norm,
        name="similarity",
        save_path=f"{output_dir}/embeddings_similarity.png",
        sim_range=sim_range,
    )
    plot_layerwise_average_change(
        self_attn_layerwise_sim,
        mlp_layerwise_sim,
        name="similarity",
        save_path=f"{output_dir}/layerwise_similarity_plots.png",
        sim_range=sim_range,
    )

    sim_range = (0.8, 1.0)
    plot_embedding_similarity(
        input_embed_dis,
        output_embed_dis,
        input_embed_avg_norm,
        output_embed_avg_norm,
        name="relative_dis",
        save_path=f"{output_dir}/embeddings_rel_dis.png",
        sim_range=sim_range,
    )
    plot_layerwise_average_change(
        self_attn_layerwise_rel_dis,
        mlp_layerwise_rel_dis,
        name="relative_dis",
        save_path=f"{output_dir}/layerwise_relative_dis_plots.png",
        sim_range=sim_range,
    )


if __name__ == "__main__":
    """
    主流程：加载模型，计算相似度，并绘图
    """

    # # 模型路径
    # base_model_path = "meta-llama/Llama-3.1-8B"
    # sft_model_path = "nvidia/OpenMath2-Llama3.1-8B"
    # # # sft_model_path = "OpenScholar/Llama-3.1_OpenScholar-8B"
    # # # sft_model_path = "/share/project/lijijie/tools/transfer_hf/Llama3_1-8B-6M-0729-megatron-fix"
    # output_dir = f'{sft_model_path.split("/")[-1]}/weight_changing_analysis'

    # analysis_parameters(base_model_path, sft_model_path, output_dir)

    base_model_path = "Qwen/Qwen2.5-7B-Instruct"
    sft_model_paths = [f"./saves/Open-Thoughts-2k-10k/Qwen2.5-7B-Instruct/full/checkpoint-936"]

    for sft_model_path in sft_model_paths:

        output_dir = f"{sft_model_path}/weight_changing_analysis"

        analysis_parameters(base_model_path, sft_model_path, output_dir)
