from collections import defaultdict
import json
import random
import zipfile
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import regex as re


def deduplicate(content):
    # idx = content.rfind('. ')
    # content = content[:idx + 1] if idx != -1 else content
    
    min_idx = 1000000
    max_postfix = None
    
    for i in range(10, len(content) // 2):
        postfix = content[-i:]
    
        idx = content[:-i].rfind(postfix)
        if idx > 0 and idx < min_idx:
            min_idx = idx
            max_postfix = postfix
        
        if idx == -1:
            break
            
    if max_postfix is not None:
        return content[:min_idx + len(max_postfix)]
    else:
        return content

def plot_single_model_metric_diff(pred_file, metric="activations_norm", layer_range=28):
    """
    分析单个模型预测结果，根据 correctness 区分 correct / wrong 样本，
    分别统计每层的 metric（如 activations_norm）的均值和方差，并绘图。
    """

    def load_predictions(pred_file):
        with open(pred_file, 'r') as f:
            return json.load(f)

    def extract_activation(activation_file):
        with zipfile.ZipFile(os.path.join('skythought', activation_file[0]), 'r') as zipf:
            with zipf.open(activation_file[1]) as file:
                return pickle.loads(file.read())

    # 收集 correct / wrong 对应的 activation metrics
    def get_metric(data, metric):

        metric_data = defaultdict(list)

        for idx, entry in data.items():
            responses = entry.get("responses", [])
            token_usages = entry.get("token_usages", [])
            activation_files = entry.get("activation_file", [])

            correct_lengths = []
            for i in range(len(responses)):
                if i >= len(token_usages):
                    continue
                if not (token_usages[i].get("completion_tokens", 0) == 16384):  # correctness = True
                    text = responses[i].get("content", "")
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    correct_lengths.append(len(encoded))
            
            if len(correct_lengths) == 0:
                # print(f"⚠️ Sample {idx} has no correct responses.")
                continue
            
            avg_prefix = int(np.mean(correct_lengths)) if len(correct_lengths) > 0 else 5000
            
            for i in range(len(responses)):
                if i >= len(token_usages) or i >= len(activation_files):
                    continue
                
                # # 忽略 exceed 情况
                # if token_usages[i].get("completion_tokens", 0) == 16384:
                #     continue

                # correctness = responses[i].get("correctness", False)
                # key = "correct" if correctness else "wrong"
                
                exceed = token_usages[i].get("completion_tokens", 0) == 16384
                key = "exceed" if exceed else "nomal"
                
                # prefix = 16384
                # if exceed:
                #     output_text = deduplicate(responses[i]['content'])
                #     prefix = len(tokenizer.encode(output_text, add_special_tokens=False))
                #     if prefix == 16384:
                #         print(output_text)
                    
                prefix = avg_prefix
                try:
                    result = extract_activation(activation_files[i])
                except Exception as e:
                    print(f"Error loading activation: {activation_files[i]} -> {e}")
                    continue

                layers = list(result.keys())[:layer_range] if layer_range > 0 else list(result.keys())[layer_range:]
                layer_metrics = []
                
                for layer in layers:
                    if metric == "activations_norm":
                        values = np.array(result[layer]['activations_norm'][:prefix])
                        diffs = values[1:] - values[:-1]
                        diffs = np.sort(diffs)[-int(prefix * 0.01):]
                        
                        layer_metrics.append(np.std(values))
                        # layer_metrics.append(np.mean(diffs))
                        
                        # # --- IQR 离群检测 ---
                        # Q1 = np.percentile(values, 25)
                        # Q3 = np.percentile(values, 75)
                        # IQR = Q3 - Q1
                        # upper = Q3 + 4 * IQR
                        # outliers = values > upper
                        # ratio = np.sum(outliers) / len(values)

                        # # --- 保存离群比例作为该层 metric ---
                        # layer_metrics.append(ratio)
                            
                    elif metric == "delta_norms":
                        values = result[layer]['delta_norms']
                        
                        rel_values = result[layer]['delta_norms'] / result[layer]['activations_norm'][:-1]
                        # values = np.sort(rel_values)[-50:]
                        
                        layer_metrics.append(np.mean(rel_values[:prefix]))
    
                metric_data[key].append(layer_metrics)
                metric_data['all'].append(layer_metrics)

        return metric_data
    
    # keys = ['correct', 'wrong']
    keys = ['exceed', 'nomal']
        
    # 加载并分析
    predictions = load_predictions(pred_file)
    metric_data = get_metric(predictions, metric)

    # 画图

    # plt.figure(figsize=(10, 6))
    # for key in keys:
    #     if not metric_data[key]:
    #         continue
    #     data = np.array(metric_data[key])
    #     mean = data.mean(axis=0)
    #     std = data.std(axis=0)
    #     plt.plot(mean, label=f'{key} (mean)')
    #     plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)

    # plt.title(f'{metric.capitalize()} by Layer (Correct vs Wrong)')
    # plt.xlabel('Layer')
    # plt.ylabel(metric)
    # plt.legend()
    # plt.grid(True)

    # save_path = f"single_model_{metric}_diff.png"
    # plt.savefig(save_path)
    # plt.show()
    # print(f"✅ 图像已保存: {save_path}")
    # plt.close()

    # 画图：correct - wrong 的差异
    plt.figure(figsize=(10, 6))

    exceed_data = np.array(metric_data[keys[0]])
    normal_data = np.array(metric_data[keys[1]])
    
    print('normal: ', normal_data.mean(axis=0).mean())
    print('exceed: ', exceed_data.mean(axis=0).mean())
    print(normal_data.mean(axis=0).mean() - exceed_data.mean(axis=0).mean())
    all_data = np.concatenate([exceed_data, normal_data], axis=0)
    print(all_data.mean(axis=0).mean())
    
    if len(exceed_data) == 0 or len(normal_data) == 0:
        print("⚠️ 缺少 correct 或 wrong 样本，跳过绘图")
        return

    correct_mean = exceed_data.mean(axis=0)
    wrong_mean = normal_data.mean(axis=0)
    diff_mean = wrong_mean - correct_mean
    # 可选：不稳定性带
    correct_std = exceed_data.std(axis=0)
    wrong_std = normal_data.std(axis=0)
    diff_std = np.sqrt(correct_std ** 2 + wrong_std ** 2)  # 差值 std 上界估计

    plt.plot(diff_mean, label=f'{keys[0]} - {keys[1]} (mean diff)', color='blue')
    plt.fill_between(range(len(diff_mean)), diff_mean - diff_std, diff_mean + diff_std,
                    color='blue', alpha=0.2, label='Std of diff (approx upper bound)')

    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'{metric} Difference by Layer: Normal - Exceed')
    plt.xlabel('Layer')
    plt.ylabel(f'Delta {metric}')
    plt.legend()
    plt.grid(True)

    plt.show()

# 示例调用

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


# metric = 'activations_norm'
metric = 'delta_norms' 
pred_files = [
    # '/global_data/pretrain/xuyao/SkyThought/merged_predictions-lora-64.json',
    # '/global_data/pretrain/xuyao/SkyThought/merged_predictions-lora-128.json',
    # '/global_data/pretrain/xuyao/SkyThought/merged_predictions-full.json',
    # '/global_data/pretrain/xuyao/SkyThought/merged_predictions-shift-v3.5-abl2.json',
    '/global_data/pretrain/xuyao/SkyThought/merged_predictions-full-olympiadbench.json',
    
]

for pred_file in pred_files:
    plot_single_model_metric_diff(pred_file, metric=metric, layer_range=28)