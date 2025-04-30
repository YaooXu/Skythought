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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_replicas=8

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# num_replicas=4


export SHIFT_RANK=64


export CHECKPOINT_SAVE='./save'

shift_versions=(
    v2cat_scale_glu_relu
)

# Evaluation tasks
tasks=(
    # "gsm8k|4"
    # "math500|4"
    # "olympiadbench_math_en|4"
    # "livecodebench|3"
    # "aime24|16"
    "aime25|16"
    # "amc23|16"
)

# 每个 config 用 “路径|rank” 的形式写
train_configs=(
    # "configs/train_full/qwen2-7b_full_sft_math_long_cot_20k-shift_gate.yaml|128"
    "configs/train_full/qwen2-7b_full_sft_math_long_cot_20k-shift_gate.yaml|256"
    # "configs/train_lora/qwen2-7b_lora_sft_math_long_cot_20k-64-shift_gate.yaml|64"
)

# 遍历每个配置
for config_item in "${train_configs[@]}"; do
    # 分割 config_path 和 rank
    IFS='|' read -r config_path rank <<< "$config_item"

    echo "Training with config: $config_path (rank=$rank)"

    config_name=$(basename "$config_path")
    config_name="${config_name%.yaml}"

    for version in "${shift_versions[@]}"; do
        export SHIFT_VERSION="${version}-${rank}"

        echo "Current SHIFT_VERSION: $SHIFT_VERSION"

        # 构建输出路径
        output_path="$CHECKPOINT_SAVE/$config_name/$SHIFT_VERSION"
        if [[ "$config_name" == *"lora"* ]]; then
            output_path="$output_path/complete_ckpt"
        fi

        # 可选：执行训练
        # FORCE_TORCHRUN=1 llamafactory-cli train "$config_path"

        # 执行评估
        for task_str in "${tasks[@]}"; do
            IFS='|' read -r task_name n <<< "$task_str"

            echo "Evaluating model: $output_path on task: $task_name (n=$n)"

            skythought evaluate \
                --model "$output_path" \
                --system-prompt-name skythought \
                --task "$task_name" \
                --backend ray \
                --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
                --sampling-params temperature=0.6,top_p=0.95,max_tokens=16384 \
                --n=$n \
                --result-dir "./evaluate_results-temp0.6-tp95/math-long-cot-20k/$task_name"
        done

        skythought evaluate \
            --model "$output_path" \
            --system-prompt-name skythought \
            --task aime24 \
            --backend ray \
            --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
            --sampling-params temperature=0.6,top_p=0.95,max_tokens=16384 \
            --n=128 \
            --result-dir "./evaluate_results-temp0.6-tp95-n128/math-long-cot-20k/aime24"

    done
done




shift_versions=(
    v2cat_scale_glu_relu
)

tasks=(
    # "gsm8k|4"
    "math500|4"
    "olympiadbench_math_en|4"
    # "livecodebench|3"
    "aime24|16"
    "aime25|16"
    "amc23|16"
)

# 每个 config 用 “路径|rank” 的形式写
train_configs=(
    "configs/train_full/qwen2-7b_full_sft_math_long_cot_20k-shift_gate.yaml|256"
)

# 遍历每个配置
for config_item in "${train_configs[@]}"; do
    # 分割 config_path 和 rank
    IFS='|' read -r config_path rank <<< "$config_item"

    echo "Training with config: $config_path (rank=$rank)"

    config_name=$(basename "$config_path")
    config_name="${config_name%.yaml}"

    for version in "${shift_versions[@]}"; do
        export SHIFT_VERSION="${version}-${rank}"

        echo "Current SHIFT_VERSION: $SHIFT_VERSION"

        # 构建输出路径
        output_path="$CHECKPOINT_SAVE/$config_name/$SHIFT_VERSION"
        if [[ "$config_name" == *"lora"* ]]; then
            output_path="$output_path/complete_ckpt"
        fi

        # 可选：执行训练
        # FORCE_TORCHRUN=1 llamafactory-cli train "$config_path"

        # Run evaluation
        for tmp in 0.2 0.4 0.6 0.8; do
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
done



