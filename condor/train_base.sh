pip install --upgrade wandb --quiet
export WANDB_API_KEY="PUT YOUR 86-CHAR OR 40-CHAR API KEY HERE"

MY_ROOT="fati/indic-language-identification"
PROJECT_ROOT="${HOME}/${MY_ROOT}"
cd "${PROJECT_ROOT}"

echo "config arg raw: $2"

echo "HOST=$(hostname)"

python3 $1 --config configs/$2
