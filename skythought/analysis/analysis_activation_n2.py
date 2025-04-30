from collections import defaultdict
import json
import zipfile
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

def extract_activation(activation_file):
    try:
        with zipfile.ZipFile(os.path.join('skythought', activation_file[0]), 'r') as zipf:
            with zipf.open(activation_file[1]) as file:
                return pickle.loads(file.read())
    except Exception as e:
        print(f"Error loading {activation_file}: {e}")
        return None

def get_sample_metrics(pred_file, metric="delta_norms", layer_range=28):
    with open(pred_file, 'r') as f:
        data = json.load(f)

    results = {}
    for sample_id, entry in data.items():
        responses = entry.get("responses", [])
        token_usages = entry.get("token_usages", [])
        activation_files = entry.get("activation_file", [])

        if not responses or not activation_files or len(responses) != len(activation_files):
            continue

        # 平均前缀长度
        correct_lengths = []
        for i in range(len(responses)):
            if i >= len(token_usages):
                continue
            if token_usages[i].get("completion_tokens", 0) < 16384:
                text = responses[i].get("content", "")
                encoded = tokenizer.encode(text, add_special_tokens=False)
                correct_lengths.append(len(encoded))

        prefix = int(np.mean(correct_lengths)) if correct_lengths else 5000

        for i in range(len(activation_files)):
            result = extract_activation(activation_files[i])
            if result is None:
                continue

            layers = list(result.keys())[:layer_range] if layer_range > 0 else list(result.keys())[layer_range:]
            metrics = []

            for layer in layers:
                if metric == "activations_norm":
                    values = np.array(result[layer]['activations_norm'][:prefix])
                    metrics.append(np.std(values))
                elif metric == "delta_norms":
                    values = result[layer]['delta_norms']
                    metrics.append(np.mean(values))
            results[sample_id] = metrics
            break  # 只处理一个 response
    return results

def plot_model_comparison(pred_files, labels, metric="delta_norms", layer_range=28):
    """
    画图比较多个模型在同一个样本上的激活特征（平均后每层）
    """
    all_model_metrics = [get_sample_metrics(f, metric=metric, layer_range=layer_range) for f in pred_files]

    # 找出所有模型都出现的样本
    common_ids = set.intersection(*[set(metrics.keys()) for metrics in all_model_metrics])
    print(f"✅ Common samples: {len(common_ids)}")

    # 每层的平均指标，模型对比
    model_layer_avgs = []

    for i, model_metrics in enumerate(all_model_metrics):
        matrix = np.array([model_metrics[sid] for sid in common_ids])
        model_layer_avgs.append(matrix.mean(axis=0))

    # 画图
    plt.figure(figsize=(10, 6))
    for label, mean in zip(labels, model_layer_avgs):
        plt.plot(mean, label=label)

    plt.title(f"Per-layer Mean {metric} (on Common Samples)")
    plt.xlabel("Layer")
    plt.ylabel(f"Mean {metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"multi_model_{metric}_compare.png"
    plt.savefig(save_path)
    print(f"✅ 图像已保存: {save_path}")
    plt.close()

# 设置模型路径和标签
pred_files = [
    '/global_data/pretrain/xuyao/SkyThought/merged_predictions-lora-64.json',
    '/global_data/pretrain/xuyao/SkyThought/merged_predictions-lora-128.json',
    '/global_data/pretrain/xuyao/SkyThought/merged_predictions-full.json',
    '/global_data/pretrain/xuyao/SkyThought/merged_predictions-shift-v3.5-abl2.json',
]
labels = ['LoRA-64', 'LoRA-128', 'Full', 'Shift-Abl2']

# 画图
plot_model_comparison(pred_files, labels, metric="delta_norms", layer_range=28)
