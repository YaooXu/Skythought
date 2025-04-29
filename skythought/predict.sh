export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export ALL_PROXY=socks5h://127.0.0.1:11300

tasks=(
    aime24
    amc23
)

model_path=save/qwen2-7b_full_sft_math_code_long_cot_20k

# model_path=save/qwen2-7b_full_sft_math_long_cot_20k-shift_gate/v2cat-256
# export SHIFT_VERSION="v2cat-256"
 
for task in "${tasks[@]}"; do
    for i in {0..7}; do
        echo "Launching job on GPU $i"
        CUDA_VISIBLE_DEVICES=$i nohup skythought evaluate \
            --model $model_path \
            --system-prompt-name skythought \
            --task $task \
            --backend vllm \
            --batch-size 1 \
            --sampling-params temperature=0.5,top_p=0.8,max_tokens=16384,seed=$i \
            --result-dir ./evaluate_results_n/$model_path/gpu_$i \
            --overwrite > logs/${model_path//\//_}-$i.log 2>&1 &
    done
done

wait

for i in {0..7}
do
    echo "Launching job on GPU $i"
    CUDA_VISIBLE_DEVICES=$i nohup skythought evaluate \
        --model saves/math-long-cot-40k/Qwen2.5-7B-Instruct/full \
        --system-prompt-name skythought \
        --task math500 \
        --backend vllm \
        --batch-size 1 \
        --sampling-params temperature=0.5,top_p=0.8,max_tokens=16384,seed=$i \
        --result-dir ./evaluate_results_with_activation_n/full/gpu_$i \
        --overwrite > logs/eval_gpu_$i.log 2>&1 &
done

# cd /global_data/pretrain/xuyao/SkyThought/skythought/
# export SHIFT_RANK=64
# export SHIFT_VERSION=v3.5-abl2
# for i in {0..7}
# do
#     echo "Launching job on GPU $i"
#     CUDA_VISIBLE_DEVICES=$i nohup skythought evaluate \
#         --model saves/math-long-cot-40k/Qwen2.5-7B-Instruct/lora-64-shift_gate/v3.5-abl2/complete_ckpt \
#         --system-prompt-name skythought \
#         --task math500 \
#         --backend vllm \
#         --batch-size 1 \
#         --sampling-params temperature=0.5,top_p=0.8,max_tokens=16384,seed=$i \
#         --result-dir ./evaluate_results_with_activation_n/shift-v3.5-abl2/gpu_$i \
#         --overwrite > logs/eval_gpu_$i.log 2>&1 &
# done