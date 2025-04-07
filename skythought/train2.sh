export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export ALL_PROXY=socks5h://127.0.0.1:11300

export VERSION=v2.3


FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29503 llamafactory-cli train configs/train_full/qwen2-7b_full_sft-bs17k-64.yaml
FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29503 llamafactory-cli train configs/train_lora/qwen2-7b_lora_sft-bs17k-64-shift_gate.yaml
FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29503 llamafactory-cli train configs/train_lora/qwen2-7b_lora_sft-bs17k-64.yaml


for task in gsm8k math500 aime24 olympiadbench_math_en ; do
    for model in \
        saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/full \
        saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/lora-64-shift_gate_$VERSION/complete_ckpt \
        saves/Bespoke-Stratos-17k/Qwen2.5-7B-Instruct/lora-64/complete_ckpt \
        ; do
        echo $task $model
        skythought evaluate \
            --model $model \
            --system-prompt-name skythought \
            --task $task \
            --backend ray \
            --backend-args tensor_parallel_size=1,num_replicas=8 \
            --result-dir ./evaluate_results/$VERSION/$task \
            --overwrite
    done
done