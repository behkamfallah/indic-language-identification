MY_ROOT="indic-language-identification"
PROJECT_ROOT="${HOME}/${MY_ROOT}"
cd "${PROJECT_ROOT}"

pip install --upgrade wandb --quiet
export WANDB_API_KEY=$(cat .wandb_api_key)

pip install matplotlib --quiet
# python3 $1 analyze_tsne.py \
#   --config task1_baseline_mms300m.yaml \
python3 $1 --config configs/$2 --split "train"
