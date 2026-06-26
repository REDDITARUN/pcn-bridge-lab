import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset

from data import VQACollator, load_vqa_dataset
from eval_binary import candidate_losses
from eval_choice import CLEVR_ANSWERS, normalize
from model import ConnectorVLM


def short_name(run_name: str) -> str:
    name = run_name.split("-", 2)[2]
    return (
        name.replace("pcn-depth3-", "pcn-")
        .replace("eqprop-momentum", "eqprop")
        .replace("mlp-depth3-", "mlp-")
    )


def load_best_checkpoints(runs_dir: Path):
    ckpts = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or "smoke" in run.name:
            continue
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.load(summary_path.open("r", encoding="utf-8"))
        ckpt = summary.get("best_checkpoint")
        if ckpt and ckpt != "None":
            ckpts.append((short_name(run.name), Path(ckpt)))
    return ckpts


def truncate(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def clean_response(text: str) -> str:
    text = " ".join(str(text).replace("\n", " ").replace("\r", " ").split())
    return text.strip()


@torch.no_grad()
def evaluate_checkpoint(label, ckpt_path, samples, device, max_new_tokens):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = ConnectorVLM(cfg).to(device)
    model.connector.load_state_dict(ckpt["connector"])
    model.eval()
    collator = VQACollator(model.image_processor, model.tokenizer)
    loader = DataLoader(samples, batch_size=len(samples), collate_fn=collator, num_workers=0)
    batch = next(iter(loader))
    pixel_values = batch["pixel_values"].to(device)
    generated = model.generate_answers(pixel_values, batch["prompts"], max_new_tokens=max_new_tokens)
    losses = [candidate_losses(model, pixel_values, batch["prompts"], c) for c in CLEVR_ANSWERS]
    pred_idx = torch.stack(losses, dim=1).argmin(dim=1).detach().cpu().tolist()
    preds = [CLEVR_ANSWERS[i] for i in pred_idx]
    rows = []
    for i, (pred, gen, answer) in enumerate(zip(preds, generated, batch["answers"])):
        gold = normalize(answer)
        rows.append({
            "sample": i,
            "model": label,
            "prediction": pred,
            "generated": clean_response(gen),
            "correct": normalize(pred) == gold,
        })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def plot_samples(samples, result_df, out_path: Path):
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(22, max(5.2 * n, 9)), gridspec_kw={"width_ratios": [1.0, 2.9]})
    if n == 1:
        axes = [axes]
    for i, sample in enumerate(samples):
        ax_img, ax_txt = axes[i]
        ax_img.imshow(sample["image"])
        ax_img.axis("off")
        ax_img.set_title(f"Sample {i + 1}")

        gt = normalize(sample["answer"])
        lines = [
            "Q: " + textwrap.fill(truncate(sample["question"], 230), width=92),
            f"Ground truth: {sample['answer']} ({gt})",
            "",
            "OK | model                  | choice | free generation",
            "---+------------------------+--------+-----------------------------------------------",
        ]
        rows = result_df[result_df["sample"] == i].sort_values("model")
        for _, row in rows.iterrows():
            mark = "OK" if row["correct"] else "--"
            gen = truncate(row["generated"], 68)
            lines.append(f"{mark:2} | {row['model']:<22} | {str(row['prediction']):<6} | {gen}")
        ax_txt.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.2)
        ax_txt.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_choices_only(samples, result_df, out_path: Path):
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(16, max(3.6 * n, 7)), gridspec_kw={"width_ratios": [1.0, 2.2]})
    if n == 1:
        axes = [axes]
    for i, sample in enumerate(samples):
        ax_img, ax_txt = axes[i]
        ax_img.imshow(sample["image"])
        ax_img.axis("off")
        gt = normalize(sample["answer"])
        rows = result_df[result_df["sample"] == i].sort_values("model")
        lines = [
            "Q: " + textwrap.fill(truncate(sample["question"], 180), width=72),
            f"GT: {gt}",
            "",
        ]
        for _, row in rows.iterrows():
            mark = "OK" if row["correct"] else "--"
            lines.append(f"{mark:2} {row['model']:<22} -> {row['prediction']}")
        ax_txt.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        ax_txt.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_summary(result_df, out_path: Path):
    summary = result_df.groupby("model", as_index=False)["correct"].sum().sort_values("correct", ascending=False)
    sns.set_theme(style="whitegrid", context="paper", palette=["#6b705c", "#a5a58d", "#b08968", "#7f5539"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(summary, x="model", y="correct", hue="model", legend=False, ax=ax)
    ax.set_title("Correct Predictions on Qualitative Validation Samples")
    ax.set_xlabel("")
    ax.set_ylabel("Correct Count")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--out", default="outputs/qualitative")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = load_best_checkpoints(Path(args.runs_dir))
    if not ckpts:
        raise SystemExit("No best checkpoints found.")
    base_cfg = torch.load(ckpts[0][1], map_location="cpu")["config"]
    ds = load_vqa_dataset(base_cfg["dataset"])["validation"]
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.n)
    samples = [ds[i] for i in indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for label, ckpt_path in ckpts:
        print(f"qualitative eval: {label}", flush=True)
        rows.extend(evaluate_checkpoint(label, ckpt_path, samples, device, args.max_new_tokens))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "qualitative_results.csv", index=False)
    sample_rows = []
    for i, idx in enumerate(indices):
        sample_rows.append({"sample": i, "val_index": idx, "question": samples[i]["question"], "answer": samples[i]["answer"]})
    pd.DataFrame(sample_rows).to_csv(out_dir / "sample_metadata.csv", index=False)
    plot_samples(samples, result_df, out_dir / "qualitative_samples.png")
    plot_choices_only(samples, result_df, out_dir / "qualitative_choices_only.png")
    plot_summary(result_df, out_dir / "qualitative_correct_counts.png")
    print(result_df.groupby("model")["correct"].sum().sort_values(ascending=False).to_string())
    print(f"Saved qualitative report to {out_dir}")


if __name__ == "__main__":
    main()
