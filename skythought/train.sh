source /root/miniconda3/bin/activate /root/miniconda3/envs/skythought

export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# export http_proxy=http://172.23.3.11:3128
# export https_proxy=http://172.23.3.11:3128

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_replicas=8

export SHIFT_RANK=128

shift_versions=(
    # "v3.5"
    # "v3.5-sub"
    "v3.5-hi-1"
)

# Evaluation tasks
tasks=(
    gsm8k
    math500
    olympiadbench_math_en
    aime24
    amc23
    livecodebench
)

# Run training and evaluation for each version
for version in "${shift_versions[@]}"; do
    export SHIFT_VERSION="$version"
    
    echo $SHIFT_VERSION
    # Training configurations with their corresponding output paths
    train_configs=(
        # "configs/train_lora/qwen2-7b_lora_sft-math-code-long_cot-64-shift_gate.yaml|saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/lora-64-shift_gate/$SHIFT_VERSION/complete_ckpt"
        "configs/train_lora/qwen2-7b_lora_sft-math-code-long_cot-128-shift_gate.yaml|saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/lora-128-shift_gate/$SHIFT_VERSION/complete_ckpt"
    )
    
    for config_pair in "${train_configs[@]}"; do
        IFS='|' read -r config_path output_path <<< "$config_pair"
        echo "Training with config: $config_path"
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
                --result-dir "./evaluate_results/Bespoke-Stratos-17k/$task"
        done
    done
done

