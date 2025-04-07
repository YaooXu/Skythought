import json
import random
import zipfile
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# prediction_files = [
#     'skythought/evaluate_results_with_activation/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_math500_055198/results.json',
#     'skythought/evaluate_results_with_activation/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_math500_b111cf/results.json',
#     # 'skythought/evaluate_results_with_activation/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5_complete_ckpt_math500_bcb298/results.json'
# ]

prediction_files = [
    'skythought/evaluate_results_with_activation/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_full_math500_2b7b0e/results.json',
    'skythought/evaluate_results_with_activation/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_math500_0afa09/results.json',
    'skythought/evaluate_results_with_activation/saves_math-long-cot-40k_Qwen2.5-7B-Instruct_lora-64-shift_gate_v3.5-abl2_complete_ckpt_math500_9c2f81/results.json'
    
    # 'skythought/evaluate_results_with_activation/saves_math-short-cot-40k_Qwen2.5-7B-Instruct_full_math500_eb9917/results.json',
    # 'skythought/evaluate_results_with_activation/saves_math-short-cot-40k_Qwen2.5-7B-Instruct_lora-64_complete_ckpt_math500_949ee3/results.json'
]

def process_activation_data(prediction_file, layer_range=28):
    print(f'Processing activation data from {prediction_file}...')
    
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)

    key1= 'cos_sims'
    key2 = 'delta_norms'
    # key2= 'relative_deltas'
    key3 = 'activations_norm'

    # Prepare lists to store average cosine similarities
    cos_sims_correct = []
    cos_sims_wrong = []

    delta_norms_correct = []
    delta_norms_wrong = []
    
    correct_len = []
    wrong_len = []
    
    c_pairs, w_paris = [], []
            
    for i in predictions:
        prediction = predictions[i]
    
        total_length = prediction['token_usages'][0]['completion_tokens']
        exceed = total_length == 16384
        if exceed:
            continue
        
        activation_file = prediction['activation_file']
    
        correctness = prediction['responses'][0]['correctness']
        
        with zipfile.ZipFile(os.path.join('skythought', activation_file[0]), 'r') as zipf:
            with zipf.open(activation_file[1]) as file:
                serialized_dict = file.read()
                result = pickle.loads(serialized_dict)
            
                current_cos_sims = []
                current_delta_norms = []
                current_activations_norm = []
                
                if layer_range > 0:
                    layers = list(result.keys())[:layer_range]
                else:
                    layers = list(result.keys())[layer_range:]
                
                length = 1000
                for layer in layers:
                    current_cos_sims.append(np.arccos(result[layer][key1][:length]).mean())
                    current_delta_norms.append(result[layer][key2][:length].mean())
                    current_activations_norm.append(result[layer][key3][:length].mean())

                # print(current_cos_sims)
                # print(current_activations_norm)
                
                # avg_cos_sim = np.mean(current_cos_sims)
                # avg_delta_norm = np.mean(current_delta_norms)
            
                if correctness:
                    cos_sims_correct.append(current_cos_sims)
                    delta_norms_correct.append(current_delta_norms)
                    correct_len.append(total_length)
                    
                    c_pairs.append((np.mean(current_cos_sims), total_length))
                else:
                    cos_sims_wrong.append(current_cos_sims)
                    delta_norms_wrong.append(current_delta_norms)
                    wrong_len.append(total_length)
                    
                    w_paris.append((np.mean(current_cos_sims), total_length))
                    
    
    # cos_sims_correct = random.sample(cos_sims_correct, 20)
    # cos_sims_wrong = random.sample(cos_sims_wrong, 20)
    # delta_norms_correct = random.sample(delta_norms_correct, 20)
    # delta_norms_wrong = random.sample(delta_norms_wrong, 20)
    
    print(np.mean(correct_len), np.mean(wrong_len))
    
    print(len(cos_sims_correct), len(cos_sims_wrong))
    cos_sims_correct_mean = np.mean(cos_sims_correct, axis=0)
    cos_sims_wrong_mean = np.mean(cos_sims_wrong, axis=0)
    delta_norms_correct_mean = np.mean(delta_norms_correct, axis=0)
    delta_norms_wrong_mean = np.mean(delta_norms_wrong, axis=0)

    cos_sims_correct_std = np.std(cos_sims_correct, axis=0)
    cos_sims_wrong_std = np.std(cos_sims_wrong, axis=0)
    delta_norms_correct_std = np.std(delta_norms_correct, axis=0)
    delta_norms_wrong_std = np.std(delta_norms_wrong, axis=0)

    # 打印均值和标准差
    print("Correct - Mean arccos:", np.mean(cos_sims_correct_mean), "Std:", np.mean(cos_sims_correct_std))
    print("Correct - Mean arccos:", np.mean(delta_norms_correct_mean), "Std:", np.mean(delta_norms_correct_std))
    print("Wrong - Mean cosine sim:", np.mean(cos_sims_wrong_mean), "Std:", np.mean(cos_sims_wrong_std))
    print("Wrong - Mean delta norms:", np.mean(delta_norms_wrong_mean), "Std:", np.mean(delta_norms_wrong_std))

    print(c_pairs)
    plt.scatter(*zip(*c_pairs), label='correct')
    plt.scatter(*zip(*w_paris), label='wrong')
    plt.show()
    
    # plt.scatter(cos_sims_correct_mean, delta_norms_correct_mean, label='correct')
    # plt.scatter(cos_sims_wrong_mean, delta_norms_wrong_mean, label='wrong')
    # plt.legend()
    # # 绘制折线图
    # plt.figure(figsize=(12, 6))

    # # 绘制余弦相似度
    # plt.subplot(1, 2, 1)
    # plt.errorbar(range(len(cos_sims_correct_mean)), cos_sims_correct_mean, yerr=cos_sims_correct_std, label='Correct', marker='o')
    # plt.errorbar(range(len(cos_sims_wrong_mean)), cos_sims_wrong_mean, yerr=cos_sims_wrong_std, label='Wrong', marker='x')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Cosine Similarity')
    # plt.title('Cosine Similarity with Variance')
    # plt.legend()

    # # 绘制Delta Norms
    # plt.subplot(1, 2, 2)
    # plt.errorbar(range(len(delta_norms_correct_mean)), delta_norms_correct_mean, yerr=delta_norms_correct_std, label='Correct', marker='o')
    # plt.errorbar(range(len(delta_norms_wrong_mean)), delta_norms_wrong_mean, yerr=delta_norms_wrong_std, label='Wrong', marker='x')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Delta Norms')
    # plt.title('Delta Norms with Variance')
    # plt.legend()

    plt.tight_layout()
    plt.show()
    
for prediction_file in prediction_files:
    process_activation_data(prediction_file, layer_range=28)