#!/bin/bash

for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=16 \
        --test_generator o1-mini \
        --lcb_version release_v2 \
        --num_round 1 \
        --no_dspy_gen \
        --selection generated_tests_majority_no_public_tests \
        --result_json_path="results/majority_4o_mini_n_16_${difficulty}.json" \

done