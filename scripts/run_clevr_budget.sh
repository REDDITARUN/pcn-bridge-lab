#!/usr/bin/env bash
set -euo pipefail

bash scripts/train_all.sh
bash scripts/train_pcn_sgd.sh
bash scripts/eval_choice_all.sh
bash scripts/analyze_all.sh
