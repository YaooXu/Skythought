import json
import pandas as pd

import json
import numpy as np

def analysis_correctness(results_filepath='skythought/evaluate_results/v3.5/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_gsm8k_439c1a/results.json'):
    with open(results_filepath, 'r') as f:
        results = json.load(f)

    all_correctness = []
    all_exceed = []
    all_length = []

    for i in results:
        result = results[i]

        length_correct = []
        length_wrong = []

        for j in range(len(result['responses'])):
            correctness = result['responses'][j]['correctness']
            total_length = result['token_usages'][j]['prompt_tokens'] + result['token_usages'][j]['completion_tokens']
            exceed = 1 if total_length in [16385, 32769] else 0
            
            # if exceed:
            #     correctness = False
            
            all_correctness.append(correctness)
            all_exceed.append(exceed)
            all_length.append(total_length)

    # 创建DataFrame
    df = pd.DataFrame({
        'correctness': all_correctness,
        'exceed': all_exceed,
        'length': all_length
    })

    # 打印基本信息
    print('Total samples:', len(df))
    print('\nError samples by exceed status:')
    # print(df[df['correctness'] == 0].groupby('exceed').size())

    print(df[df['correctness'] == 1].groupby('exceed').size())

    # 计算并打印正确和错误样本的平均长度
    avg_length_correct = df[df['correctness'] == 1]['length'].mean()
    avg_length_error = df[df['correctness'] == 0]['length'].mean()
    
    print(f'\nAverage length of CORRECT samples: {avg_length_correct:.1f} tokens')
    print(f'Average length of ERROR samples: {avg_length_error:.1f} tokens')

    return df


pathes = [
    # '/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results-temp0.7/math-long-cot-20k/math500/._save_qwen2-7b_lora_sft_math_long_cot_20k-64_complete_ckpt_math500_4fb95e/results.json',
    # '/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results-temp0.7/math-long-cot-20k/math500/._save_qwen2-7b_lora_sft_math_long_cot_20k-64-shift_gate_v2cat-64_complete_ckpt_math500_698453/results.json'
    
    # '/mnt/workspace/user/sunwangtao/Skythought/skythought/diff_temps/evaluate_results-temp0.6-tp95/math-long-cot-20k/olympiadbench_math_en/._save_qwen2-7b_full_sft_math_long_cot_20k_olympiadbench_math_en_c19d03/results.json',
    # '/mnt/workspace/user/sunwangtao/Skythought/skythought/diff_temps/evaluate_results-temp0.6-tp95/math-long-cot-20k/olympiadbench_math_en/._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_scale_glu_relu-256_olympiadbench_math_en_7b4bd6/results.json'
    '/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results/temp0.6-tp95/math-long-cot-20k-32768/amc23/._save_qwen2-7b_lora_sft_math_long_cot_20k-128_complete_ckpt_amc23_f8f3f7/results.json',
    '/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results/temp0.6-tp95/math-long-cot-20k-32768/amc23/._save_qwen2-7b_lora_sft_math_long_cot_20k-128-shift_gate_v2cat_scale_glu_relu-128_complete_ckpt_amc23_1bf3fd/results.json',
    '/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results/temp0.6-tp95/math-long-cot-20k-32768/amc23/._save_qwen2-7b_lora_sft_math_long_cot_20k-256_complete_ckpt_amc23_4f572f/results.json',
    '/mnt/workspace/user/sunwangtao/Skythought/skythought/evaluate_results/temp0.6-tp95/math-long-cot-20k-32768/amc23/._save_qwen2-7b_lora_sft_math_long_cot_20k-256-shift_gate_v2cat_scale_glu_relu-256_complete_ckpt_amc23_860358/results.json'
    
]

for p in pathes:
    print(p)
    analysis_correctness(p)
    print('---')