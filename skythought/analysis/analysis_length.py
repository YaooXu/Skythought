import json
import pandas as pd

import json
import pandas as pd

def analysis_correctness(results_filepath='skythought/evaluate_results/v3.5/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_gsm8k_439c1a/results.json'):
    with open(results_filepath, 'r') as f:
        results = json.load(f)

    all_correctness = []
    all_exceed = []
    all_length = []

    for result in results:
        length_correct = []
        length_wrong = []
    
        for i in range(len(result['responses'])):
            correctness = result['responses'][i]['correctness']
            total_length = result['token_usages'][i]['prompt_tokens'] + result['token_usages'][0]['completion_tokens']
            exceed = 1 if total_length == 16385 else 0
            
            if exceed:
                continue
            
            if correctness:
                length_correct.append(total_length)
            else:
                length_wrong.append(total_length)

        print('correct', np.mean(length_correct))
        print('wrong', np.mean(length_wrong))
        
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
    # # GSM8K
    # 'skythought/evaluate_results/v3.5/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_gsm8k_439c1a/results.json',
    # 'skythought/evaluate_results/v3.5/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_gsm8k_ed15bd/results.json',
    # 'skythought/evaluate_results/v3.5/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5_complete_ckpt_gsm8k_307e89/results.json',
    # 'skythought/evaluate_results/openthoughts-math-40k/v3.5-abl2/gsm8k/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5-abl2_complete_ckpt_gsm8k_1bb1aa/results.json'
    
    # Math500
    'skythought/evaluate_results/openthoughts-math-40k/v3.5/math500/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_math500_05b119/results.json',
    'skythought/evaluate_results/openthoughts-math-40k/v3.5/math500/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_math500_c2fc4c/results.json',
    # 'skythought/evaluate_results/v3.5/math500/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5_complete_ckpt_math500_0933bd/results.json',
    # 'skythought/evaluate_results/openthoughts-math-40k/v3.5-abl2/math500/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5-abl2_complete_ckpt_math500_d9b1b7/results.json'
    
    # # olympiadbench_math_en
    # 'skythought/evaluate_results/v3.5/olympiadbench_math_en/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_olympiadbench_math_en_e30860/results.json',
    # 'skythought/evaluate_results/v3.5/olympiadbench_math_en/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_olympiadbench_math_en_4776d8/results.json',
    # 'skythought/evaluate_results/v3.5/olympiadbench_math_en/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5_complete_ckpt_olympiadbench_math_en_baf5f8/results.json',
    # 'skythought/evaluate_results/openthoughts-math-40k/v3.5-abl2/olympiadbench_math_en/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5-abl2_complete_ckpt_olympiadbench_math_en_53d9b5/results.json'
    
    # # amc23
    # 'skythought/evaluate_results/v3.5/amc23/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_amc23_e99a68/results.json',
    # 'skythought/evaluate_results/v3.5/amc23/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_amc23_be9506/results.json',
    # 'skythought/evaluate_results/v3.5/amc23/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5_complete_ckpt_amc23_5806e6/results.json',
    # 'skythought/evaluate_results/openthoughts-math-40k/v3.5-abl2/amc23/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5-abl2_complete_ckpt_amc23_8987ae/results.json'
    
    # # aime24
    # 'skythought/evaluate_results/v3.5/aime24/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_aime24_153f74/results.json',
    # 'skythought/evaluate_results/v3.5/aime24/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_aime24_dbe2c3/results.json',
    # 'skythought/evaluate_results/v3.5/aime24/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5_complete_ckpt_aime24_c83c22/results.json',
    # 'skythought/evaluate_results/openthoughts-math-40k/v3.5-abl2/aime24/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5-abl2_complete_ckpt_aime24_23dec5/results.json'
]

for p in pathes:
    print(p)
    analysis_correctness(p)
    print('---')