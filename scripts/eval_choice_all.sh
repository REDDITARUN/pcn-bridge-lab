#!/usr/bin/env bash
set -euo pipefail

for summary in outputs/runs/*/summary.json; do
  ckpt=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("best_checkpoint", ""))' "${summary}")
  if [[ -n "${ckpt}" && "${ckpt}" != "None" ]]; then
    python src/eval_choice.py --checkpoint "${ckpt}" --split test
  fi
done
