# source /root/miniconda3/bin/activate /root/miniconda3/envs/skythought
# export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface

export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# export http_proxy=http://172.23.3.11:3128
# export https_proxy=http://172.23.3.11:3128
# export ALL_PROXY=socks5://127.0.0.1:11300

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_replicas=8


export CHECKPOINT_SAVE='./save'

# Evaluation tasks
tasks=(
    gsm8k
    math500
    olympiadbench_math_en
    aime24
    amc23
    livecodebench
)

train_configs=(
    "configs/train_lora/qwen2-7b_lora_sft_math_code_short_cot_20k-32.yaml"
    "configs/train_lora/qwen2-7b_lora_sft_math_code_short_cot_20k-64.yaml"
    "configs/train_lora/qwen2-7b_lora_sft_math_code_short_cot_20k-128.yaml"
    "configs/train_full/qwen2-7b_full_sft_math_code_short_cot.yaml"

    "configs/train_lora/qwen2-7b_lora_sft_math_code_long_cot_20k-32.yaml"
    "configs/train_lora/qwen2-7b_lora_sft_math_code_long_cot_20k-64.yaml"
    "configs/train_lora/qwen2-7b_lora_sft_math_code_long_cot_20k-128.yaml"
    "configs/train_full/qwen2-7b_full_sft_math_code_long_cot.yaml"
)

for config_path in "${train_configs[@]}"; do
    echo "Training with config: $config_path"

    config_name=$(basename "$config_path")
    output_path="$CHECKPOINT_SAVE/$config_name"

    if [[ "$config_name" == *"lora"* ]]; then
    output_path="$output_path/complete_ckpt"
    fi

    echo "Output will be saved to: $output_path"
    
    FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29503 llamafactory-cli train "$config_path"

    # Run evaluation
    for task in "${tasks[@]}"; do
        echo "Evaluating model: $output_path on task: $task"
        skythought evaluate \
            --model "$output_path" \
            --system-prompt-name skythought \
            --task "$task" \
            --backend ray \
            --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
            --sampling-params temperature=0.7,max_tokens=16384 \
            --result-dir "./evaluate_results/math-code-long-cot-20k/$task"
    done
done
