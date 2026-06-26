# Experiment History

This file preserves prior pilot findings before generated outputs are cleared for new controlled runs.

## Pilot 1: M3IT Shapes + SmolLM2-135M

Setup:

- Vision encoder: `openai/clip-vit-base-patch32`, frozen
- Language model: `HuggingFaceTB/SmolLM2-135M`, frozen
- Dataset: `MMInstruction/M3IT`, `shapes`
- Trainable module: connector only
- Connectors: Linear, 3-layer MLP, 3-layer PCN
- PCN settle steps: `1`, `6`, `12`
- PCN optimizers: AdamW, SGD momentum, EqPropMomentum-style CE-nudged physics update

Main findings:

- PCN AdamW had the best answer CE losses.
- Best validation CE was PCN 1-step AdamW, around `0.1383`.
- Test binary accuracy was near the dataset majority baseline because the test set was about `63% no`.
- Free-form generation was messy with SmolLM2-135M.

Conclusion:

- Useful as an engineering smoke test, but not a strong scientific benchmark.
- The LM was too weak and the dataset too biased for a defensible PCN-vs-MLP claim.

## Pilot 2: CLEVR + Qwen2.5-0.5B-Instruct

Setup:

- Vision encoder: `openai/clip-vit-base-patch32`, frozen
- Language model: `Qwen/Qwen2.5-0.5B-Instruct`, frozen
- Dataset source: `dpdl-benchmark/clevr`, flattened image-question-answer samples
- Train/eval subset: `5k` train, `500` eval
- Epochs: `3`
- Trainable module: connector only
- Connectors: Linear, 3-layer MLP, 3-layer PCN
- PCN settle steps: `1`, `6`, `12`
- PCN optimizers: AdamW, SGD momentum, EqPropMomentum-style CE-nudged physics update

Result summary by CLEVR candidate-choice accuracy:

| Rank | Run | Choice Accuracy |
|---:|---|---:|
| 1 | PCN 1-step AdamW | `46.0%` |
| 2 | PCN 12-step EqPropMomentum | `45.2%` |
| 3 | MLP AdamW | `44.0%` |
| 4 | PCN 1-step EqPropMomentum | `43.0%` |
| 5 | PCN 1-step SGD | `42.4%` |
| 6 | PCN 6-step AdamW | `42.2%` |
| 7 | PCN 12-step AdamW | `40.6%` |
| 8 | PCN 12-step SGD | `39.6%` |
| 9 | Linear AdamW | `38.4%` |
| 10 | PCN 6-step EqPropMomentum | `31.2%` |
| 11 | PCN 6-step SGD | `30.2%` |

Result summary by best validation CE:

| Rank | Run | Best Validation Loss |
|---:|---|---:|
| 1 | PCN 1-step AdamW | `0.9500` |
| 2 | Linear AdamW | `0.9545` |
| 3 | PCN 12-step AdamW | `0.9644` |
| 4 | PCN 6-step SGD | `0.9680` |
| 5 | PCN 12-step SGD | `0.9694` |
| 6 | PCN 6-step AdamW | `0.9729` |
| 7 | PCN 12-step EqPropMomentum | `0.9781` |
| 8 | PCN 1-step SGD | `0.9786` |
| 9 | PCN 1-step EqPropMomentum | `0.9848` |
| 10 | PCN 6-step EqPropMomentum | `0.9892` |
| 11 | MLP AdamW | `1.0094` |

Main findings:

- PCN 1-step AdamW was the cleanest pilot winner by both validation CE and choice accuracy.
- PCN 12-step EqPropMomentum was close by choice accuracy, suggesting EqPropMomentum may be worth further tuning.
- More settling steps did not consistently help; `1` step beat `6` and `12` for AdamW.
- MLP was competitive, but not parameter-matched against PCN.

Important caveat:

- MLP had `6.69M` trainable connector parameters, while PCN had `9.65M`.
- Because PCN had about `44%` more trainable parameters, the pilot was not capacity-controlled.

## Next Controlled Run

Motivation:

- Make the PCN-vs-MLP comparison more defensible by parameter-matching MLP to PCN.
- Use more data and more epochs while keeping the run bounded.

Planned setup:

- Dataset: `dpdl-benchmark/clevr`
- Train samples: `10k`
- Eval/test samples: `1k`
- Epochs: `5`
- Frozen CLIP: same
- Frozen Qwen2.5-0.5B-Instruct: same
- Train connector only: same
- MLP hidden size: `1070`, giving about `9.646M` params, matching PCN at about `9.646M`
- Full matrix: Linear, matched MLP, PCN AdamW `1/6/12`, PCN SGD `1/6/12`, PCN EqPropMomentum `1/6/12`

## Controlled Run Invalidated: Qwen Chat Template Issue

What happened:

- A controlled CLEVR/Qwen run was completed with `10k` train samples, `1k` eval/test samples, `5` epochs, and parameter-matched MLP/PCN connectors.
- The qualitative CSV showed malformed free generations such as `Human#1`, `Human Answer:`, repeated prompt fragments, and mixed-language artifacts.
- Inspection showed that `Qwen/Qwen2.5-0.5B-Instruct` was being prompted with a plain text prompt instead of Qwen's native chat template.

Why this matters:

- Qwen Instruct models are trained to consume chat-formatted prompts produced by `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- The old prompt format likely made generation and candidate scoring noisier than necessary.
- Therefore, the previous controlled results should be treated as invalidated and not used as final evidence.

Fix applied:

- `src/data.py` now formats prompts with `tokenizer.apply_chat_template` when the tokenizer has a chat template.
- The chat prompt uses a system instruction: `Answer the visual question with exactly one word or one number. Do not explain.`
- Choice eval and qualitative eval already consume prompts from `VQACollator`, so they now inherit the corrected chat template automatically.
- `src/eval_binary.py` now scores binary candidates as `yes` and `no`, matching the short-answer instruction.

Status:

- Old generated results were deleted and should be regenerated from scratch with the corrected prompt format.

## Corrected Controlled Run: CLEVR + Qwen Chat Template

Setup:

- Vision encoder: `openai/clip-vit-base-patch32`, frozen
- Language model: `Qwen/Qwen2.5-0.5B-Instruct`, frozen
- Prompting: Qwen chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
- Dataset source: `dpdl-benchmark/clevr`, flattened image-question-answer samples
- Train samples: `10k`
- Eval/test samples: `1k`
- Epochs: `5`
- Trainable module: connector only
- Connectors: Linear, parameter-matched 3-layer MLP, 3-layer PCN
- MLP params: `9.645728M`
- PCN params: `9.646336M`
- PCN settle steps: `1`, `6`, `12`
- PCN optimizers: AdamW, SGD momentum, EqPropMomentum-style CE-nudged physics update
- Primary metric: CLEVR candidate-choice accuracy over valid answer candidates

Result summary by candidate-choice accuracy:

| Rank | Run | Choice Accuracy | Best Val CE |
|---:|---|---:|---:|
| 1 | PCN 1-step EqPropMomentum | `47.2%` | `0.9762` |
| 2 | PCN 1-step AdamW | `46.3%` | `0.9662` |
| 3 | PCN 6-step AdamW | `45.9%` | `0.9610` |
| 4 | PCN 6-step SGD | `45.5%` | `0.9737` |
| 5 | PCN 12-step EqPropMomentum | `45.5%` | `0.9741` |
| 6 | PCN 1-step SGD | `45.2%` | `0.9747` |
| 7 | Linear AdamW | `45.0%` | `0.9577` |
| 8 | PCN 12-step SGD | `45.0%` | `0.9706` |
| 9 | MLP AdamW | `43.8%` | `0.9698` |
| 10 | PCN 6-step EqPropMomentum | `43.2%` | `0.9812` |
| 11 | PCN 12-step AdamW | `43.0%` | `0.9759` |

Qualitative free-generation check:

- A second qualitative pass with `max_new_tokens=32` showed that free generations are still noisy despite the chat-template fix.
- Examples included generic text such as `Human beings are...`, `Humanity`, and unrelated problem/answer fragments.
- Because of this, free generation should be treated as diagnostic only, not as the primary evidence.

Main findings:

- The best corrected run was PCN 1-step EqPropMomentum at `47.2%` candidate-choice accuracy.
- The best PCN exceeded the parameter-matched MLP by `3.4` absolute percentage points.
- Several PCN variants exceeded matched MLP, so the controlled result supports a PCN advantage under this setup.
- More settling steps did not consistently help; the best model used `1` settle step.
- EqPropMomentum was best at `1` step but weaker at `6` steps, so optimizer benefits are not uniform.
- Linear AdamW remained competitive at `45.0%`, so future work should include multiple seeds and stronger baselines before making broad claims.

Publication notes:

- No obvious private tokens or credentials were found in the project files or result CSV/JSON outputs.
- Do not publish `outputs/cache/` or `src/__pycache__/`.
- For Hugging Face, prefer publishing stripped connector-only weights rather than full checkpoints with optimizer state.
