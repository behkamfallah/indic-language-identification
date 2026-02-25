# pip install --upgrade wandb --quiet
# export WANDB_API_KEY="PUT YOUR 86-CHAR OR 40-CHAR API KEY HERE"

MY_ROOT="fati/indic-language-identification"
PROJECT_ROOT="${HOME}/${MY_ROOT}"
cd "${PROJECT_ROOT}"

# python3 $1 analyze_tsne.py \
#   --config task1_baseline_mms300m.yaml \
python3 $1 \
    --config configs/$2 \
  # --model_dir ./models/<YOUR_BASELINE_RUN_ID_FOLDER> \
  # --split eval \
  # --out_dir ./outputs/tsne_baseline