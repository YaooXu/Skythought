

# Shift-FFN

This is the official implement of Shift-FFN, this repository is based on
[SkyThought](https://github.com/NovaSky-AI/SkyThought), we use the same training and evaluation settings as SkyThought.




# Getting Started

```shell
# prepare environment
# python3.10, cuda124, torch251, flash_attn274

git clone https://github.com/YaooXu/Skythought
cd Skythought

pip install -r requirements.txt
pip install -e .

cd LoRA-GA
pip install -e peft

cd ../skythought/train/LLaMA-Factory
pip install -e .

cd ./src/llamafactory/vllm_add_shift_models
pip install -e .

pip uninstall transformer-engine
```

```shell
# prepare data

python scripts/process_openthoughts_metadata.py

cd skythought
bash run_all.sh

```

# Evaluation

## Usage

You can install the latest release from PyPI or from [source](#installing-from-source):

Running evaluation is as simple as: 

```bash
skythought evaluate --model NovaSky-AI/Sky-T1-32B-Preview --task aime24
```

This checkpoint of Qwen with Shift-FFN will come soon.
