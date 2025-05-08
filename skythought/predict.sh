# export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export ALL_PROXY=socks5h://127.0.0.1:11300

tasks=(
    math500
    # olympiadbench_math_en
    # aime24
    # amc23
)


export SHIFT_VERSION="v2cat_scale_glu_relu-64"
model_path=(
    # save/qwen2-7b_lora_sft_math_long_cot_20k-64-shift_gate/v2cat_scale_glu_relu-64/complete_ckpt
    save/qwen2-7b_lora_sft_math_long_cot_20k-64/complete_ckpt
    save/qwen2-7b_lora_sft_math_long_cot_20k-128/complete_ckpt
    save/qwen2-7b_full_sft_math_long_cot_20k
)
 
for task in "${tasks[@]}"; do
    for model in "${model_path[@]}"; do
        echo $task $model
        for i in {0..7}; do
            echo "Launching job on GPU $i"
            CUDA_VISIBLE_DEVICES=$i nohup skythought evaluate \
                --model $model \
                --system-prompt-name skythought \
                --task $task \
                --backend vllm \
                --batch-size 1 \
                --sampling-params temperature=0.6,top_p=0.95,max_tokens=16384,seed=$i \
                --result-dir ./evaluate_results_n-temp0.6-tp95-h/$task/gpu_$i \
                --overwrite > logs/${model//\//_}-$i.log 2>&1 &
        done

        wait

    done
done
