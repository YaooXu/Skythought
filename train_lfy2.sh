# git pull
# cd LoRA-GA
# git pull
# cd ..

export HF_ENDPOINT=https://hf-mirror.com

# python scripts/process_openthoughts_metadata.py

cd skythought

export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_replicas=8

# chekpoint保存路径
export CHECKPOINT_SAVE='./save'

# Evaluation tasks
tasks=(
    "math500|4"
    "olympiadbench_math_en|4"
    "aime24|32"
    "aime25|32"
    "amc23|32"

    "aime24|128"
    "aime25|128"
    "amc23|128"
    "math500|128"
    "olympiadbench_math_en|128"

)


# shift model
shift_versions=(
    v2cat_scale_glu_relu
)

train_configs=(
    "configs/train_lora_lfy/qwen2-7b_lora_sft_math_long_cot_40k-256.yaml"
    "configs/train_lora_lfy/qwen2-7b_lora_sft_math_long_cot_80k-296.yaml"
    "configs/train_lora_lfy/qwen2-7b_lora_sft_math_long_cot_80k-256-shift_gate.yaml|256" # exit
)

# 遍历每个配置
for config_item in "${train_configs[@]}"; do
    # 检查config_path是否包含"gate"
    if [[ "$config_path" == *"gate1.6"* ]]; then
        export GATE_RANK_COE="1.636" # qwen 7b
    else
        export GATE_RANK_COE="1"
    fi


    # 分割 config_path 和 rank
    IFS='|' read -r config_path rank <<< "$config_item"

    echo "Training with config: $config_path (rank=$rank)"

    config_name=$(basename "$config_path")
    config_name="${config_name%.yaml}"

    size_part=$(echo "$config_name" | grep -oE '[0-9]+k')

    for version in "${shift_versions[@]}"; do
        export SHIFT_VERSION="${version}-${rank}"

        echo "Current SHIFT_VERSION: $SHIFT_VERSION"

        # 构建输出路径
        output_path="$CHECKPOINT_SAVE/$config_name/$SHIFT_VERSION"
        if [[ "$config_name" == *"lora"* ]]; then
            output_path="$output_path/complete_ckpt"
        fi

        # Check if output directory exists
        if [ ! -d "$output_path" ]; then
            echo "Directory $output_path doesn't exist. Starting training..."
            
            export HF_ENDPOINT=https://hf-mirror.com
            FORCE_TORCHRUN=1 /cpfs01/data/shared/Group-m6/fangyu.lfy/conda_env/sky/bin/llamafactory-cli train "$config_path"

        else
            echo "Directory $output_path exists. Skipping training."
        fi

        # 执行评估
        for task_str in "${tasks[@]}"; do
            IFS='|' read -r task_name n <<< "$task_str"

            echo "Evaluating model: $output_path on task: $task_name (n=$n)"

            export HF_ENDPOINT=https://hf-mirror.com
            cpfs01/data/shared/Group-m6/fangyu.lfy/conda_env/sky/bin/skythought evaluate \
                --model "$output_path" \
                --system-prompt-name skythought \
                --task "$task_name" \
                --backend ray \
                --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
                --sampling-params temperature=0.6,top_p=0.95,max_tokens=16384 \
                --n=$n \
                --result-dir "./evaluate_results/temp0.6-tp95/math-long-cot-$size_part/$task_name"
        done
    done
done



# # base model
# train_configs=(
#     "configs/train_full_lfy/qwen2-7b_full_sft_math_long_cot_80k.yaml"
# )

# export GATE_RANK_COE="1"

# for config_path in "${train_configs[@]}"; do
#     echo "Training with config: $config_path"

#     config_name=$(basename "$config_path")
#     config_name="${config_name%.yaml}"
#     output_path="$CHECKPOINT_SAVE/$config_name"
    
#     # 提取数字部分（40k或80k）
#     size_part=$(echo "$config_name" | grep -oE '[0-9]+k')
    
#     if [[ "$config_name" == *"lora"* ]]; then
#         output_path="$output_path/complete_ckpt"
#     fi

#     echo "Output will be saved to: $output_path"
    
#     export HF_ENDPOINT=https://hf-mirror.com
#     FORCE_TORCHRUN=1 /cpfs01/data/shared/Group-m6/fangyu.lfy/conda_env/sky/bin/llamafactory-cli train "$config_path"

#     # Run evaluation
#     for task_str in "${tasks[@]}"; do
#         IFS='|' read -r task_name n <<< "$task_str"

#         echo "Evaluating model: $output_path on task: $task_name (n=$n)"

#         export HF_ENDPOINT=https://hf-mirror.com
#         cpfs01/data/shared/Group-m6/fangyu.lfy/conda_env/sky/bin/skythought evaluate \
#             --model "$output_path" \
#             --system-prompt-name skythought \
#             --task "$task_name" \
#             --backend ray \
#             --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
#             --sampling-params temperature=0.6,top_p=0.95,max_tokens=16384 \
#             --n=$n \
#             --result-dir "./evaluate_results/temp0.6-tp95/math-long-cot-$size_part/$task_name"
#     done
# done

# # base model
# train_configs=(
#     "configs/train_full_lfy/qwen2-7b_full_sft_math_long_cot_40k.yaml"
# )

# export GATE_RANK_COE="1"

# for config_path in "${train_configs[@]}"; do
#     echo "Training with config: $config_path"

#     config_name=$(basename "$config_path")
#     config_name="${config_name%.yaml}"
#     output_path="$CHECKPOINT_SAVE/$config_name"
    
#     # 提取数字部分（40k或80k）
#     size_part=$(echo "$config_name" | grep -oE '[0-9]+k')
    
#     if [[ "$config_name" == *"lora"* ]]; then
#         output_path="$output_path/complete_ckpt"
#     fi

#     echo "Output will be saved to: $output_path"
    
#     # export HF_ENDPOINT=https://hf-mirror.com
#     # FORCE_TORCHRUN=1 /cpfs01/data/shared/Group-m6/fangyu.lfy/conda_env/sky/bin/llamafactory-cli train "$config_path"

#     # Run evaluation
#     for task_str in "${tasks[@]}"; do
#         IFS='|' read -r task_name n <<< "$task_str"

#         echo "Evaluating model: $output_path on task: $task_name (n=$n)"

#         export HF_ENDPOINT=https://hf-mirror.com
#         cpfs01/data/shared/Group-m6/fangyu.lfy/conda_env/sky/bin/skythought evaluate \
#             --model "$output_path" \
#             --system-prompt-name skythought \
#             --task "$task_name" \
#             --backend ray \
#             --backend-args "tensor_parallel_size=1,num_replicas=$num_replicas" \
#             --sampling-params temperature=0.6,top_p=0.95,max_tokens=16384 \
#             --n=$n \
#             --result-dir "./evaluate_results/temp0.6-tp95/math-long-cot-$size_part/$task_name"
#     done
# done
