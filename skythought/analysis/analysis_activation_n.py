from collections import defaultdict
import json
import random
import zipfile
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import regex as re

os.environ['HF_HOME']='mnt/workspace/user/sunwangtao/.cache/huggingface'
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
                length = token_usages[i]['completion_tokens'] + token_usages[i]['prompt_tokens']
                if length in [16384, 16385, 32769]:
                    exceed = 1
                else:
                    exceed = 0

                if not exceed:
                    text = responses[i].get("content", "")
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    correct_lengths.append(len(encoded))
            
            if len(correct_lengths) == 0:
                # print(f"⚠️ Sample {idx} has no correct responses.")
                continue
            
            prefix = int(np.mean(correct_lengths)) if len(correct_lengths) > 0 else 16384

            for i in range(len(responses)):
            
                length = token_usages[i]['completion_tokens'] + token_usages[i]['prompt_tokens']
                if length in [16384, 16385, 32769]:
                    exceed = 1
                else:
                    exceed = 0

                key = "exceed" if exceed else "nomal"
                
                if exceed:
                    # print(prefix)
                    output_text = deduplicate(responses[i]['content'])
                    prefix = min(prefix, len(tokenizer.encode(output_text, add_special_tokens=False)))
                    print(prefix, len(tokenizer.encode(output_text, add_special_tokens=False)))
                    print('\n\n')
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
                            
                    elif metric == "delta_norms":
                        values = result[layer]['delta_norms']
                        
                        rel_values = result[layer]['relative_deltas']
                        
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
    exceed_data = np.array(metric_data[keys[0]])
    normal_data = np.array(metric_data[keys[1]])

    # 计算每个样本在所有层上的均值 (shape: [n_samples,])
    exceed_means = exceed_data.mean(axis=1)  # 沿层方向求均值
    normal_means = normal_data.mean(axis=1)

    all_means = np.concatenate([exceed_means, normal_means])
    print(f'All (mean over layers): {all_means.mean():.4f}')
    print(len(exceed_means) / len(all_means), len(normal_means) / len(all_means))

    # 打印统计信息
    print(f'Normal (mean over layers): {normal_means.mean():.4f}')
    print(f'Exceed (mean over layers): {exceed_means.mean():.4f}')
    # print(f'Mean diff (Normal - Exceed): {normal_means.mean() - exceed_means.mean():.4f}')

    if len(exceed_means) == 0 or len(normal_means) == 0:
        print("⚠️ 缺少 exceed 或 normal 样本，跳过绘图")
        return

    # # ---------- 新增：从 normal 中随机选取与 exceed 相同数量的样本 ----------
    # n_exceed = len(exceed_means)
    # if n_exceed < len(normal_means):
    #     # 随机选取（不重复抽样）
    #     random_normal_means = np.random.choice(normal_means, size=n_exceed, replace=False)
    # else:
    #     # 如果 exceed 样本数比 normal 多，则全部选取（避免报错）
    #     random_normal_means = normal_means.copy()


    # ---------- 绘制箱线图 ----------
    plt.figure(figsize=(3, 3.5))  # 保持较窄的宽度

    # 三组数据：exceed, normal (full), normal (random subset)
    box_data = [exceed_means, normal_means, all_means]
    labels = [
        f'Exceed',
        f'Normal',
        f'All'
    ]

    # 关键修改：调整positions参数减少箱线间距
    positions = [1, 1.8, 2.6]  # 原始默认是[1,2,3]，这里缩小间距

    # 计算各组平均数
    means = [np.mean(data) for data in box_data]

    # 画箱线图
    boxplot = plt.boxplot(
        box_data,
        positions=positions,  # 使用自定义位置
        labels=labels,
        patch_artist=True,
        widths=0.5,  # 保持较窄的箱体宽度
        showmeans=True,
        meanline=True,
        meanprops={'linestyle': '--', 'linewidth': 1.5, 'color': 'darkred'},
        medianprops={'color': 'black', 'linewidth': 1.5},
        showfliers=False,
    )

    # 自定义颜色
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加平均数标注
    # for pos, mean in zip(positions, means):
    #     plt.text(pos, mean+0.05*(plt.ylim()[1]-plt.ylim()[0]),  # 位置稍微上移
    #             f'{mean:.2f}', 
    #             ha='center', va='bottom',
    #             fontsize=8,
    #             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.ylabel(f'Mean relative change (over layers)', fontsize=9)
    plt.xticks(positions, labels, fontsize=8)  # 确保标签对应新位置
    plt.yticks(fontsize=8)
    # plt.ylim(0.73, 0.83)
    plt.grid(axis='y', linestyle=':', alpha=0.5)

    # 调整x轴范围使图形更紧凑
    plt.xlim(positions[0]-0.5, positions[-1]+0.5)

    plt.tight_layout()
    plt.savefig('boxplot_comparison.pdf', bbox_inches='tight', dpi=500)
    plt.show()

# 示例调用

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


# metric = 'activations_norm'
metric = 'delta_norms' 
pred_files = [
    'h-merged_predictions-save_qwen2-7b_full_sft_math_long_cot_20k-math500.json',

    # '/mnt/workspace/user/sunwangtao/Skythought/merged_predictions-qwen2-7b_full_sft_math_long_cot_20k-math500.json',
    # '/mnt/workspace/user/sunwangtao/Skythought/merged_predictions-qwen2-7b_lora_sft_math_long_cot_20k-64-shift_gate_v2cat_scale_glu_relu-64_complete_ckpt-math500.json',
    # '/mnt/workspace/user/sunwangtao/Skythought/h-merged_predictions-save_qwen2-7b_lora_sft_math_long_cot_20k-64_complete_ckpt-math500.json'
]

for pred_file in pred_files:
    plot_single_model_metric_diff(pred_file, metric=metric, layer_range=28)