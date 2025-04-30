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
    "math500|8"
    "olympiadbench_math_en|8"
    "aime24|16"
    "aime25|16"
    "amc23|16"
)


train_configs=(
    # "configs/train_lora/qwen2-7b_lora_sft_math_long_cot_20k-32.yaml"
    # "configs/train_lora/qwen2-7b_lora_sft_math_long_cot_20k-64.yaml"
    # "configs/train_lora/qwen2-7b_lora_sft_math_long_cot_20k-128.yaml"
    "configs/train_full/qwen2-7b_full_sft_math_long_cot_20k.yaml"
    
    # "configs/train_lora/qwen2-7b_lora_sft_math_short_cot_20k-32.yaml"
    # "configs/train_lora/qwen2-7b_lora_sft_math_short_cot_20k-64.yaml"
    # "configs/train_lora/qwen2-7b_lora_sft_math_short_cot_20k-128.yaml"
    # "configs/train_full/qwen2-7b_full_sft_math_short_cot_20k.yaml"
)

for config_path in "${train_configs[@]}"; do
    echo "Training with config: $config_path"

    config_name=$(basename "$config_path")
    config_name="${config_name%.yaml}"
    output_path="$CHECKPOINT_SAVE/$config_name"
    
    if [[ "$config_name" == *"lora"* ]]; then
        output_path="$output_path/complete_ckpt"
    fi

    echo "Output will be saved to: $output_path"
    
    # FORCE_TORCHRUN=1 llamafactory-cli train "$config_path"

    # Run evaluation
    for tmp in 0.6 ; do
        for task_str in "${tasks[@]}"; do

            IFS='|' read -r task_name n <<< "$task_str"

            echo "Evaluating model: $output_path on task: $task_name (n=$n)"

            skythought evaluate \
                --model "$output_path" \
                --system-prompt-name skythought \
                --task "$task_name" \
                --backend ray \
                --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
                --sampling-params temperature=$tmp,top_p=0.95,max_tokens=16384 \
                --n=$n \
                --result-dir "./diff_temps/evaluate_results-temp$tmp-tp95/math-long-cot-20k/$task_name"
        done
    done

done
