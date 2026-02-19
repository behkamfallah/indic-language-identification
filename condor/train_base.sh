MY_ROOT="fati/indic-language-identification"
PROJECT_ROOT="${HOME}/${MY_ROOT}"
cd "${PROJECT_ROOT}"

echo "config arg raw: $2"

export CUDA_VISIBLE_DEVICES=""

python3 $1 --config configs/$2
