# pip install --upgrade wandb --quiet
# export WANDB_API_KEY="PUT YOUR 86-CHAR OR 40-CHAR API KEY HERE"

MY_ROOT="fati/indic-language-identification"
PROJECT_ROOT="${HOME}/${MY_ROOT}"
cd "${PROJECT_ROOT}"

pip install matplotlib --quiet
# python3 $1 analyze_tsne.py \
#   --config task1_baseline_mms300m.yaml \
python3 $1 --config configs/$2