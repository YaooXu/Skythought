# git pull

# pip install deepspeed==0.15.3

# python scripts/process_openthoughts_metadata.py

# cd skythought

# export HF_TOKEN=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
export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT=long-cot
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# chekpoint保存路径
export CHECKPOINT_SAVE='./save'

len=32768

# len=16384

# Evaluation tasks
tasks=(
    "math500|4"
    "olympiadbench_math_en|4"
    "aime24|16"
    "amc23|16"

    "gpqa_diamond|4"
    "livecodebench|4"
    # "aime25|32"

    # "aime24|128"
)

temps=(
    # 0.2
    # 0.4
    0.6
    # 0.8
    # 1.0
)

size_part=114k

model_path=Qwen/Qwen3-8B
tp=1
num_replicas=8

# 执行评估
for temp in "${temps[@]}"; do
    for task_str in "${tasks[@]}"; do
        IFS='|' read -r task_name n <<< "$task_str"

        echo "Evaluating model: $model_path on task: $task_name (n=$n)"

        skythought evaluate \
            --model "$model_path" \
            --system-prompt-name skythought \
            --task "$task_name" \
            --backend ray \
            --backend-args "tensor_parallel_size=$tp,num_replicas=$num_replicas" \
            --sampling-params temperature=$temp,top_p=0.95,max_tokens=$len \
            --n=$n \
            --result-dir "./evaluate_results/temp$temp-tp95/math-long-cot-$size_part-$len/$task_name"
    done
done

model_path=Qwen/Qwen3-1.7B
tp=1
num_replicas=8

# 执行评估
for temp in "${temps[@]}"; do
    for task_str in "${tasks[@]}"; do
        IFS='|' read -r task_name n <<< "$task_str"

        echo "Evaluating model: $model_path on task: $task_name (n=$n)"

        skythought evaluate \
            --model "$model_path" \
            --system-prompt-name skythought \
            --task "$task_name" \
            --backend ray \
            --backend-args "tensor_parallel_size=$tp,num_replicas=$num_replicas" \
            --sampling-params temperature=$temp,top_p=0.95,max_tokens=$len \
            --n=$n \
            --result-dir "./evaluate_results/temp$temp-tp95/math-long-cot-$size_part-$len/$task_name"
    done
done


export VLLM_WORKER_MULTIPROC_METHOD=spawn
model_path=Qwen/Qwen3-30B-A3B
tp=4
num_replicas=2


# 执行评估
for temp in "${temps[@]}"; do
    for task_str in "${tasks[@]}"; do
        IFS='|' read -r task_name n <<< "$task_str"

        echo "Evaluating model: $model_path on task: $task_name (n=$n)"

        skythought evaluate \
            --model "$model_path" \
            --system-prompt-name skythought \
            --task "$task_name" \
            --backend ray \
            --backend-args "tensor_parallel_size=$tp,num_replicas=$num_replicas" \
            --sampling-params temperature=$temp,top_p=0.95,max_tokens=$len \
            --n=$n \
            --result-dir "./evaluate_results/temp$temp-tp95/math-long-cot-$size_part-$len/$task_name"
    done
done


export VLLM_WORKER_MULTIPROC_METHOD=spawn
model_path=Qwen/Qwen3-32B
tp=4
num_replicas=2


# 执行评估
for temp in "${temps[@]}"; do
    for task_str in "${tasks[@]}"; do
        IFS='|' read -r task_name n <<< "$task_str"

        echo "Evaluating model: $model_path on task: $task_name (n=$n)"

        skythought evaluate \
            --model "$model_path" \
            --system-prompt-name skythought \
            --task "$task_name" \
            --backend ray \
            --backend-args "tensor_parallel_size=$tp,num_replicas=$num_replicas" \
            --sampling-params temperature=$temp,top_p=0.95,max_tokens=$len \
            --n=$n \
            --result-dir "./evaluate_results/temp$temp-tp95/math-long-cot-$size_part-$len/$task_name"
    done
done