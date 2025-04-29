# Model Performance Across Datasets

| Model | aime24 | amc23 | gsm8k | math500 | olympiadbench_math_en | Average |
| - | - | - | - | - | - | - |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat-256_ | 17.7 | 54.1 | 89.6 | 76.6 | 40.4 | 55.7 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_glu-256_ | 16.5 | 51.9 | 90.1 | 76.8 | 40.6 | 55.2 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_glu_silu-256_ | 17.1 | 53.3 | 90.3 | 75.6 | 40.2 | 55.3 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_scale-256_ | 18.5 | 54.7 | 90.5 | 77.0 | 40.5 | 56.2 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_scale-512_ | 16.7 | 54.5 | 90.0 | 76.8 | 39.8 | 55.6 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_scale_glu_relu-256_ | 16.7 | 54.5 | 90.3 | 77.3 | 40.0 | 55.8 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2cat_scale_glu_relu-512_ | 16.2 | 54.5 | 90.6 | 76.6 | 40.6 | 55.7 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2pre-256_ | - | - | 90.5 | - | - | 90.5 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2pre_glu-256_ | 17.5 | 53.4 | 90.3 | 77.2 | 40.4 | 55.8 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v2pre_glu_silu-256_ | 18.3 | 51.9 | 90.3 | 76.3 | 40.0 | 55.4 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v4cat-256_ | 16.9 | 53.0 | 90.7 | 75.7 | 39.6 | 55.2 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v4cat_scale-256_ | 16.2 | 55.0 | 90.2 | 77.0 | 39.9 | 55.7 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v4cat_scale_glu-256_ | 14.8 | 53.3 | 90.2 | 76.2 | 40.2 | 54.9 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v4pre-256_ | 16.5 | 54.5 | 90.0 | 76.8 | 40.6 | 55.7 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k-shift_gate_v4pre_glu-256_ | 17.9 | 53.9 | 90.8 | 76.4 | 40.1 | 55.8 |
| ._save_qwen2-7b_full_sft_math_long_cot_20k_ | 16.9 | 53.8 | 90.0 | 76.1 | 39.6 | 55.3 |
