# export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export ALL_PROXY=socks5h://127.0.0.1:11300

tasks=(
    # math500
    olympiadbench_math_en
    # aime24
    # amc23
)


export USE_EAGER=1
export INDEX_RANGE="0_50"

model_path=(
    save/qwen2-7b_full_sft_math_long_cot_20k
    save/qwen2-7b_lora_sft_math_long_cot_20k-64/complete_ckpt
    save/qwen2-7b_lora_sft_math_long_cot_20k-128/complete_ckpt
    # save/qwen2-7b_lora_sft_math_long_cot_20k-256/complete_ckpt
    # save/qwen2-7b_lora_sft_math_long_cot_20k-256-shift_gate/v3cat_scale_glu_relu-256/complete_ckpt
    # save/qwen2-7b_lora_sft_math_long_cot_20k-296/complete_ckpt
)

mkdir logs

echo $INDEX_RANGE

for task in "${tasks[@]}"; do
    mkdir logs/$task

    for model in "${model_path[@]}"; do
        echo $task $model
        if [[ "$model" == *shift_gate* ]]; then
            IFS='/' read -ra parts <<< "$model"
            SHIFT_VERSION="${parts[-2]}"
            export SHIFT_VERSION
            echo "Detected 'shift_gate', exporting SHIFT_VERSION=$SHIFT_VERSION"
        else
            unset SHIFT_VERSION
            echo "No 'shift_gate' detected, unsetting SHIFT_VERSION"
        fi

        base_dir=./evaluate_results_n-temp0.6-tp95-h-$INDEX_RANGE/$task/gpu_0/
        prefix=${model//\//_}_$task

        found_dirs=$(find "$base_dir" -maxdepth 1 -type d -name "${prefix}"*)
        
        if [ -f "$found_dirs"/results.json ]; then 
            echo "Results already exist for $task and model $model"
        else
            echo "Launching job for $task and model $model"

            for i in {0..7}; do
                echo "Launching job on GPU $i"
                CUDA_VISIBLE_DEVICES=$i nohup skythought evaluate \
                    --model $model \
                    --system-prompt-name skythought \
                    --task $task \
                    --backend vllm \
                    --batch-size 1 \
                    --sampling-params temperature=0.6,top_p=0.95,max_tokens=32768,seed=$i \
                    --result-dir ./evaluate_results_n-temp0.6-tp95-h-$INDEX_RANGE/$task/gpu_$i \
                    --overwrite > logs/$task/${model//\//_}-$i.log 2>&1 &
            done

            wait
        fi
    done
done
