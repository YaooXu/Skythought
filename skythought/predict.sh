export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export ALL_PROXY=socks5h://127.0.0.1:11300

export CUDA_VISIBLE_DEVICES=2
skythought evaluate \
    --model saves/math-short-cot-40k/Qwen2.5-7B-Instruct/full \
    --system-prompt-name skythought \
    --task math500 \
    --backend vllm \
    --batch-size 1 \
    --sampling-params max_tokens=16384 \
    --result-dir ./evaluate_results_with_activation \
    --overwrite

export CUDA_VISIBLE_DEVICES=5
skythought evaluate \
    --model saves/math-short-cot-40k/Qwen2.5-7B-Instruct/lora-64/complete_ckpt \
    --system-prompt-name skythought \
    --task math500 \
    --backend vllm \
    --batch-size 1 \
    --sampling-params max_tokens=16384 \
    --result-dir ./evaluate_results_with_activation \
    --overwrite

export CUDA_VISIBLE_DEVICES=6
export SHIFT_VERSION=v3.5-abl2
skythought evaluate \
    --model saves/math-long-cot-40k/Qwen2.5-7B-Instruct/lora-64-shift_gate/v3.5-abl2/complete_ckpt \
    --system-prompt-name skythought \
    --task math500 \
    --backend vllm \
    --batch-size 1 \
    --sampling-params max_tokens=16384 \
    --result-dir ./evaluate_results_with_activation \
    --overwrite
    