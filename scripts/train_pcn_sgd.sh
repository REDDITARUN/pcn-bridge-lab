#!/usr/bin/env bash
set -euo pipefail

for steps in 1 6 12; do
  python src/train.py \
    --config configs/m3it_shapes_smol.yaml \
    --name "pcn-depth3-s${steps}-sgd" \
    --override connector.type=pcn \
    --override connector.depth=3 \
    --override connector.settle_steps="${steps}" \
    --override training.optimizer=sgd \
    --override training.lr=0.01 \
    --override training.weight_decay=0.0001
done
