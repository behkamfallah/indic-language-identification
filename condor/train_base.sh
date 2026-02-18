MY_ROOT="fati/indic-language-identification"
PROJECT_ROOT="${HOME}/${MY_ROOT}"
cd "${PROJECT_ROOT}"

echo "config arg raw: $2"

python3 $1 --config configs/$2
