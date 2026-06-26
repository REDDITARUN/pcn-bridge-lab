#!/usr/bin/env bash
set -euo pipefail

for steps in 1 6 12; do
  python src/train.py \
    --config configs/m3it_shapes_smol.yaml \
    --name "pcn-depth3-s${steps}-adamw" \
    --override connector.type=pcn \
    --override connector.depth=3 \
    --override connector.settle_steps="${steps}" \
    --override training.optimizer=adamw
done

for steps in 1 6 12; do
  python src/train.py \
    --config configs/m3it_shapes_smol.yaml \
    --name "pcn-depth3-s${steps}-eqprop-momentum" \
    --override connector.type=pcn \
    --override connector.depth=3 \
    --override connector.settle_steps="${steps}" \
    --override training.optimizer=eqprop_momentum \
    --override training.lr=0.0005 \
    --override training.grad_accum_steps=1 \
    --override eqprop.beta=0.1 \
    --override eqprop.lambda_settle=0.01 \
    --override eqprop.momentum=0.9
done
