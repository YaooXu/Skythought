import json
import glob

def merge_predictions(prediction_files, output_file):
    merged = {}
    keys_to_merge = {"responses", "token_usages", "activation_file"}

    for file in prediction_files:
        with open(file, 'r') as f:
            data = json.load(f)

        for idx, entry in data.items():
            entry['activation_file'] = [tuple(entry.get('activation_file', []))]
            if idx not in merged:
                # 初始化：复制全部 key，初始化可合并的 key 为列表
                merged[idx] = {}
                for k, v in entry.items():
                    if k in keys_to_merge:
                        merged[idx][k] = list(v) if isinstance(v, list) else []
                    else:
                        merged[idx][k] = v
            else:
                # 已存在：只合并指定的 key
                for k in keys_to_merge:
                    merged[idx].setdefault(k, [])
                    merged[idx][k].extend(entry.get(k, []))

    # 按 index 排序
    merged_sorted = {k: merged[k] for k in sorted(merged, key=lambda x: int(x))}
    with open(output_file, 'w') as f:
        json.dump(merged_sorted, f, indent=2)

    print(f"✅ 合并完成，输出文件: {output_file}")

# 示例调用
task = 'math500'
name = 'save_qwen2-7b_full_sft_math_long_cot_20k'
# name = 'save_qwen2-7b_lora_sft_math_long_cot_20k-256-shift_gate_v3cat_scale_glu_relu-256'

prediction_files = glob.glob(f'skythought/evaluate_results_n-temp0.6-tp95-h-f50/{task}/**/*{name}*/results.json', recursive=True)

print(prediction_files)

merge_predictions(prediction_files, f'h-merged_predictions-{name}-{task}.json')
