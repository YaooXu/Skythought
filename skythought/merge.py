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
# task = 'math500'
# task = 'aime24'
task = 'olympiadbench_math_en'
# task = 'amc23'

names = [
    'save_qwen2-7b_full_sft_math_long_cot_20k',
    'save_qwen2-7b_lora_sft_math_long_cot_20k-64_complete_ckpt',
    'save_qwen2-7b_lora_sft_math_long_cot_20k-128_complete_ckpt',
    'save_qwen2-7b_lora_sft_math_long_cot_20k-256_complete_ckpt',
    # 'save_qwen2-7b_lora_sft_math_long_cot_20k-296_complete_ckpt',
    'save_qwen2-7b_lora_sft_math_long_cot_20k-256-shift_gate_v3cat_scale_glu_relu-256_complete_ckpt'
]

idx_ranges = [
    '0_50',
    '100_150'
]

dirs = [
    f"evaluate_results_n-temp0.6-tp95-h-{idx_range}" for idx_range in idx_ranges
]
# dir = "evaluate_results_n-temp0.6-tp95-h-100_150"

for name in names:
    prediction_files = [list(glob.glob(f'skythought/{dir}/{task}/**/*{name}_{task}*/results.json', recursive=True)) for dir in dirs]
    prediction_files = [item for sublist in prediction_files for item in sublist]

    print(prediction_files)

    merge_predictions(prediction_files, f"{'-'.join(idx_ranges)}-{name}-{task}.json")
