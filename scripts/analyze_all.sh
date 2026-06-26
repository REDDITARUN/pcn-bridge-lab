#!/usr/bin/env bash
set -euo pipefail

python src/plot.py --runs outputs/runs/* --out outputs/plots
python src/analyze.py --runs-dir outputs/runs --out outputs/analysis
python src/qualitative.py --runs-dir outputs/runs --out outputs/qualitative --n 8 --seed 17 --max-new-tokens 4
