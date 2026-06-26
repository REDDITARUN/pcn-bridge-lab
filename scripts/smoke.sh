#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --config configs/m3it_shapes_smol.yaml
  --override dataset.max_train_samples=4
  --override dataset.max_eval_samples=4
  --override training.epochs=1
  --override training.batch_size=2
  --override training.num_workers=0
  --override training.log_every=1
)

python src/train.py "${COMMON[@]}" --name smoke-mlp --override connector.type=mlp
python src/train.py "${COMMON[@]}" --name smoke-pcn-adamw --override connector.type=pcn --override connector.settle_steps=1 --override training.optimizer=adamw
python src/train.py "${COMMON[@]}" --name smoke-pcn-sgd --override connector.type=pcn --override connector.settle_steps=1 --override training.optimizer=sgd --override training.lr=0.01 --override training.weight_decay=0.0001
python src/train.py "${COMMON[@]}" --name smoke-pcn-eqprop --override connector.type=pcn --override connector.settle_steps=1 --override training.optimizer=eqprop_momentum --override training.lr=0.0005
