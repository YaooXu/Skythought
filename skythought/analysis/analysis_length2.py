import json
import pandas as pd

import json
import numpy as np

def analysis_correctness(results_filepath='skythought/evaluate_results/v3.5/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_gsm8k_439c1a/results.json'):
    with open(results_filepath, 'r') as f:
        results = json.load(f)

    all_correctness_lengths = []
    all_wrong_lengths = []

    for idx in results:
        result = results[idx]
        length_correct = []
        length_wrong = []
    
        for i in range(len(result['responses'])):
            correctness = result['responses'][i]['correctness']
            total_length = result['token_usages'][i]['prompt_tokens'] + result['token_usages'][i]['completion_tokens']
            exceed = 1 if total_length == 16385 else 0
            
            if exceed:
                continue
            
            if correctness:
                length_correct.append(total_length)
                all_correctness_lengths.append(total_length)
                
            else:
                length_wrong.append(total_length)
                all_wrong_lengths.append(total_length)

        if len(length_correct) == 0 or len(length_wrong) == 0:
            continue
        
        # print('correct', np.mean(length_correct))
        # print('wrong', np.mean(length_wrong))
        # print('------')

    print('------' * 10)
    print('correct', np.mean(all_correctness_lengths))
    print('wrong', np.mean(all_wrong_lengths))
    print('------')
    
    # # 创建DataFrame
    # df = pd.DataFrame({
    #     'correctness': all_correctness,
    #     'exceed': all_exceed,
    #     'length': all_length
    # })

    # # 打印基本信息
    # print('Total samples:', len(df))
    # print('\nError samples by exceed status:')
    # print(df[df['correctness'] == 0].groupby('exceed').size())

    # # 计算并打印正确和错误样本的平均长度
    # avg_length_correct = df[df['correctness'] == 1]['length'].mean()
    # avg_length_error = df[df['correctness'] == 0]['length'].mean()
    
    # print(f'\nAverage length of CORRECT samples: {avg_length_correct:.1f} tokens')
    # print(f'Average length of ERROR samples: {avg_length_error:.1f} tokens')

    # return df


pathes = [
    # Math500
    'skythought/evaluate_results/Bespoke-Stratos-17k/math500/saves_Bespoke-Stratos-17k_Qwen2.5-7B-Instruct_full_math500_408c01/results.json',
    'skythought/evaluate_results/Bespoke-Stratos-17k/math500/saves_Bespoke-Stratos-17k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_math500_d19ecb/results.json',
]

for p in pathes:
    print(p)
    analysis_correctness(p)
    print('---')