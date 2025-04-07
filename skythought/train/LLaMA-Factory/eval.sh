
for task in math500 aime24 gsm8k olympiadbench_math_en; do
    for model in \
        saves/Open-Thoughts-2k-10k/Qwen2.5-7B-Instruct/lora/checkpoint-936 \
        saves/Open-Thoughts-2k-10k/Qwen2.5-7B-Instruct/full/checkpoint-936 \
        saves/Open-Thoughts-16k-10k/Qwen2.5-7B-Instruct/lora/checkpoint-936 \
        saves/Open-Thoughts-16k-10k/Qwen2.5-7B-Instruct/full/checkpoint-936 \
        ; do
        echo $task $model
        skythought evaluate \
            --model $model \
            --system-prompt-name skythought \
            --task $task \
            --backend ray \
            --backend-args tensor_parallel_size=1,num_replicas=8 \
            --result-dir ./evaluate_results/$task
    done
done

# skythought evaluate \
#     --model saves/Open-Thoughts-16k-10k/Qwen2.5-7B-Instruct/lora/checkpoint-936 \
#     --system-prompt-name skythought \
#     --task math500 \
#     --backend ray \
#     --backend-args tensor_parallel_size=1,num_replicas=8 \
#     --result-dir ./evaluate_results



# CUDA_VISIBLE_DEVICES=4,5,6,7 skythought evaluate \
#     --model skythought/train/LLaMA-Factory/saves/Open-Thoughts-2k-10k/Qwen2.5-7B-Instruct/full/checkpoint-936 \
#     --task math500 \
#     --backend ray \
#     --backend-args tensor_parallel_size=1,num_replicas=4 \
#     --result-dir ./
