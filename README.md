# PCN Bridge Lab

Connector-only experiment for comparing a standard MLP bridge against a predictive-coding bridge between frozen CLIP and frozen Qwen2.5 on CLEVR.

This is an AI-assisted research/engineering experiment. The experiment design, code, runs, and documentation were produced with substantial AI assistance and a small human-in-the-loop workflow for direction, review, and decisions.

Repositories:

- Code and results: https://github.com/REDDITARUN/pcn-bridge-lab
- Connector-only weights: https://huggingface.co/Teen-Different/pcn-bridge-lab-connectors

## Objective

Test whether a 3-layer Predictive Coding Network connector aligns image features to a small language model better than a 3-layer MLP connector.

Fixed components:

- Vision encoder: `openai/clip-vit-base-patch32`
- Language model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `dpdl-benchmark/clevr`, flattened image-question-answer subset
- Frozen parameters: CLIP and Qwen2.5
- Trainable parameters: connector only
- Controlled run budget: `10k` train examples, `1k` eval examples, `5` epochs
- CLEVR subset is cached once under `outputs/cache/clevr` and reused by later runs
- MLP is parameter-matched to PCN with `connector.hidden_dim=1070` in `scripts/train_mlp.sh`

## Experiment Matrix

Main runs:

- Linear connector sanity check
- 3-layer MLP connector
- 3-layer PCN connector with settle steps `1`, `6`, `12`
- PCN optimizer ablation with AdamW, SGD momentum, and CE-nudged EqPropMomentum

Primary metrics:

- Validation cross entropy
- Test candidate-choice accuracy over valid CLEVR answers
- Train/validation curves
- Time per epoch
- Peak GPU memory
- Trainable connector parameter count

Prompting:

- Qwen prompts are formatted with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- The system instruction is: `Answer the visual question with exactly one word or one number. Do not explain.`
- Free-form generations are logged for qualitative inspection, but the main comparison uses candidate-choice scoring because generations can drift into non-answer text.

## Setup

```bash
cd /home/ubuntu/pcn-bridge-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use the published connector weights, download them from https://huggingface.co/Teen-Different/pcn-bridge-lab-connectors and place the selected connector file wherever convenient. The checkpoint files contain connector weights only; CLIP and Qwen are loaded from their original Hugging Face model IDs.

## Smoke Test

Use a tiny subset before launching full A6000 runs:

```bash
python src/train.py \
  --config configs/m3it_shapes_smol.yaml \
  --name smoke-mlp \
  --override connector.type=mlp \
  --override dataset.max_train_samples=8 \
  --override dataset.max_eval_samples=8 \
  --override training.epochs=1 \
  --override training.batch_size=2 \
  --override training.num_workers=0
```

## Test Existing Checkpoints

After training, evaluate all best checkpoints with candidate-choice scoring:

```bash
bash scripts/eval_choice_all.sh
```

Regenerate plots and qualitative reports:

```bash
bash scripts/analyze_all.sh
```

The main result table is written to:

```text
outputs/analysis/results.csv
```

## Full Runs

```bash
bash scripts/run_clevr_budget.sh
```

Or run subsets:

```bash
bash scripts/train_mlp.sh
bash scripts/train_pcn.sh
bash scripts/train_pcn_sgd.sh
```

Run a linear sanity baseline:

```bash
python src/train.py \
  --config configs/m3it_shapes_smol.yaml \
  --name linear-adamw \
  --override connector.type=linear
```

Evaluate all checkpoints with CLEVR candidate-choice scoring:

```bash
bash scripts/eval_choice_all.sh
```

Plot curves after runs finish:

```bash
python src/plot.py --runs outputs/runs/* --out outputs/plots
```

## PCN Implementation Notes

The PCN connector is intentionally not a normal feed-forward MLP with a different name. It maintains latent states, predicts lower states from higher states, computes local prediction errors, and updates latent states for multiple settling steps before producing LM visual-token embeddings.

Default PCN training uses answer-token cross entropy plus a small prediction-energy auxiliary loss. This keeps the main connector comparison fair because linear, MLP, and PCN all optimize the same answer objective.

`EqPropMomentum` is active for PCN runs when `training.optimizer=eqprop_momentum`. The training loop computes a free PCN energy, then a clamped energy nudged by answer cross entropy: `clamped_energy = pcn_energy + beta * answer_ce`. This gives a practical EqPropMomentum-style optimizer ablation for the connector while still reporting the same answer CE and candidate-choice metrics as AdamW and SGD.

## Current Corrected Results

Corrected controlled run:

- Train samples: `10k`
- Eval/test samples: `1k`
- Epochs: `5`
- Frozen models: CLIP ViT-B/32 and Qwen2.5-0.5B-Instruct
- Trainable module: connector only
- MLP and PCN trainable parameters: about `9.646M` each
- Evaluation: CLEVR candidate-choice accuracy with chat-template prompts

| Rank | Run | Choice Accuracy |
|---:|---|---:|
| 1 | PCN 1-step EqPropMomentum | `47.2%` |
| 2 | PCN 1-step AdamW | `46.3%` |
| 3 | PCN 6-step AdamW | `45.9%` |
| 4 | PCN 6-step SGD | `45.5%` |
| 5 | PCN 12-step EqPropMomentum | `45.5%` |
| 6 | PCN 1-step SGD | `45.2%` |
| 7 | Linear AdamW | `45.0%` |
| 8 | PCN 12-step SGD | `45.0%` |
| 9 | MLP AdamW | `43.8%` |
| 10 | PCN 6-step EqPropMomentum | `43.2%` |
| 11 | PCN 12-step AdamW | `43.0%` |

Main observation: the best PCN connector outperforms the parameter-matched MLP on CLEVR candidate-choice accuracy in this run. Free-form qualitative generations remain noisy and should not be treated as the primary metric.

## Outputs

Each run writes:

- `config.json`
- `metrics.jsonl`
- `checkpoint-epoch*.pt`
- `summary.json`

Plots use a minimal white-grid seaborn style with earthy colors and are written to `outputs/plots`.

Full training checkpoints are intentionally excluded from Git because they include optimizer state. For model sharing, publish stripped connector-only weights.

## License

This project is released under the MIT License. See `LICENSE`.
