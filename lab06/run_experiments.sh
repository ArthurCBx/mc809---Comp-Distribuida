#!/usr/bin/env bash
# Roda todos os experimentos do lab; os CSVs ficam em results/<experimento>/
set -e
cd "$(dirname "$0")"

ROUNDS=20

for alpha in 0.1 1 100; do
    python simulation.py --rounds $ROUNDS --fraction-fit 0.5 --alpha $alpha
done

python simulation.py --rounds $ROUNDS --fraction-fit 0.5 --iid

for alpha in 0.1 1 100; do
    python simulation.py --rounds $ROUNDS --fraction-fit 1 --alpha $alpha
done
