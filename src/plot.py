import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


EARTH = ["#6b705c", "#a5a58d", "#b08968", "#7f5539", "#3a5a40", "#bc6c25"]


def load_metrics(run_dir: Path) -> pd.DataFrame:
    rows = []
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return pd.DataFrame()
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["run"] = run_dir.name
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_metrics(Path(p)) for p in args.runs]
    frames = [x for x in frames if not x.empty]
    if not frames:
        raise SystemExit("No metrics.jsonl files found in the provided run directories.")
    df = pd.concat(frames, ignore_index=True)
    sns.set_theme(style="whitegrid", context="paper", palette=EARTH)
    plt.rcParams.update({"figure.dpi": 160, "axes.edgecolor": "#d8d2c4", "grid.color": "#ece6d8"})

    epoch_df = df[df["phase"].isin(["train_epoch", "validation"])].copy()
    if not epoch_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.lineplot(epoch_df, x="epoch", y="loss", hue="run", style="phase", marker="o", ax=ax)
        ax.set_title("Connector Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross Entropy Loss")
        sns.despine(fig)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curves.png")

    setup_df = df[df["phase"] == "setup"].copy()
    if not setup_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(setup_df, x="run", y="connector_params", hue="run", legend=False, ax=ax)
        ax.set_title("Trainable Connector Parameters")
        ax.set_xlabel("")
        ax.set_ylabel("Parameters")
        ax.tick_params(axis="x", rotation=25)
        sns.despine(fig)
        fig.tight_layout()
        fig.savefig(out_dir / "parameter_counts.png")


if __name__ == "__main__":
    main()
