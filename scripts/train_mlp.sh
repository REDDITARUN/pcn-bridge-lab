#!/usr/bin/env bash
set -euo pipefail

python src/train.py \
  --config configs/m3it_shapes_smol.yaml \
  --name mlp-depth3-adamw \
  --override connector.type=mlp \
  --override connector.depth=3 \
  --override connector.hidden_dim=1070 \
  --override training.optimizer=adamw
