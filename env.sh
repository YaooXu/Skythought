
conda activate skythought

cd /global_data/pretrain/xuyao/ss
bash restart.sh

cd /global_data/pretrain/xuyao/SkyThought/skythought/
bash train.sh


conda activate skythought
cd skythought



# export HF_HOME=/global_data/pretrain/xuyao/.cache/huggingface
# export WANDB_API_KEY=efe05a42b8b37cb8028408410c02bcefbddf42c0
# export HF_DATASETS_OFFLINE=1

# export ALL_PROXY=socks5h://127.0.0.1:11300