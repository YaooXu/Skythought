source /root/miniconda3/bin/activate /root/miniconda3/envs/skythought

export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export ALL_PROXY=socks5h://127.0.0.1:11300

# export http_proxy=http://172.23.3.11:3128
# export https_proxy=http://172.23.3.11:3128

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_replicas=8

export SHIFT_VERSION=v3.5

export SHIFT_RANK=64

tasks=(
    # gsm8k
    # math500
    # olympiadbench_math_en
    # aime24
    # amc23
    livecodebench
)

models=(
    # saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/full
    # saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/lora-64/complete_ckpt
    saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/lora-64-shift_gate/v3.5/complete_ckpt
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Evaluating model: $model_path on task: $task"
        skythought evaluate \
            --model $model \
            --system-prompt-name skythought \
            --task "$task" \
            --backend ray \
            --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
            --result-dir "./evaluate_results/Bespoke-Stratos-17k/$task"
    done
done

