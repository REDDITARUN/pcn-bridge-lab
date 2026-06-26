#!/usr/bin/env bash
set -euo pipefail

python src/train.py \
  --config configs/m3it_shapes_smol.yaml \
  --name linear-adamw \
  --override connector.type=linear \
  --override training.optimizer=adamw

bash scripts/train_mlp.sh
bash scripts/train_pcn.sh
