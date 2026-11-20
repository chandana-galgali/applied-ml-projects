#!/usr/bin/env bash
# Re-runs every project and rebuilds the website data. Usage: bash run_all.sh
set -e
cd "$(dirname "$0")"
for p in 01-house-price-regression 02-customer-segmentation 03-cats-vs-dogs-svm \
         04-hand-gesture-recognition 05-food-recognition-calories; do
  echo "================ $p"
  (cd "$p" && python -W ignore train.py)
done
python scripts/build_site.py
