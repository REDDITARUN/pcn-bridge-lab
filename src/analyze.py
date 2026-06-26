import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


EARTH = ["#6b705c", "#a5a58d", "#b08968", "#7f5539", "#3a5a40", "#bc6c25", "#606c38"]


def load_json(path: Path):
    return json.load(path.open("r", encoding="utf-8"))


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or "smoke" in run.name:
            continue
        metrics_path = run / "metrics.jsonl"
        config_path = run / "config.json"
        summary_path = run / "summary.json"
        if not metrics_path.exists() or not config_path.exists() or not summary_path.exists():
            continue
        cfg = load_json(config_path)
        summary = load_json(summary_path)
        metrics = [json.loads(line) for line in metrics_path.open("r", encoding="utf-8")]
        setup = next((m for m in metrics if m.get("phase") == "setup"), {})
        vals = [m for m in metrics if m.get("phase") == "validation"]
        trains = [m for m in metrics if m.get("phase") == "train_epoch"]
        best_val = min(vals, key=lambda x: x.get("loss", float("inf"))) if vals else {}
        ckpt = Path(summary.get("best_checkpoint", ""))
        if not str(ckpt) or str(ckpt) == "None":
            continue
        choice_path = ckpt.with_suffix(".test.choice_metrics.json") if ckpt else None
        test_path = ckpt.with_suffix(".test.metrics.json") if ckpt else None
        choice = load_json(choice_path) if choice_path and choice_path.exists() else {}
        test = load_json(test_path) if test_path and test_path.exists() else {}
        rows.append({
            "run": run.name.split("-", 2)[2],
            "connector": cfg["connector"]["type"],
            "optimizer": cfg["training"]["optimizer"],
            "settle_steps": cfg["connector"].get("settle_steps"),
            "params_m": setup.get("connector_params", 0) / 1_000_000,
            "best_val_loss": best_val.get("loss"),
            "best_epoch": best_val.get("epoch"),
            "test_loss": test.get("loss"),
            "choice_accuracy": choice.get("choice_accuracy"),
            "choice_loss": choice.get("choice_loss"),
            "avg_epoch_min": (sum(t.get("seconds", 0) for t in trains) / max(len(trains), 1)) / 60,
            "peak_mem_gb": max([m.get("peak_memory_gb", 0) for m in metrics] or [0]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_col = "choice_accuracy" if "choice_accuracy" in df.columns else "best_val_loss"
    return df.sort_values(sort_col, ascending=False if sort_col == "choice_accuracy" else True)


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", palette=EARTH)
    plt.rcParams.update({"figure.dpi": 180, "axes.edgecolor": "#d8d2c4", "grid.color": "#ece6d8"})
    for y, title, filename in [
        ("choice_accuracy", "Test CLEVR Choice Accuracy", "test_choice_accuracy.png"),
        ("best_val_loss", "Best Validation Loss", "best_validation_loss.png"),
        ("test_loss", "Test Answer CE Loss", "test_loss.png"),
        ("avg_epoch_min", "Average Epoch Time", "epoch_time.png"),
        ("peak_mem_gb", "Peak GPU Memory", "peak_memory.png"),
    ]:
        plot_df = df.dropna(subset=[y]).copy()
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.barplot(plot_df, x="run", y=y, hue="connector", dodge=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        sns.despine(fig)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_runs(Path(args.runs_dir))
    if df.empty:
        raise SystemExit("No completed non-smoke runs found.")
    df.to_csv(out_dir / "results.csv", index=False)
    df.to_json(out_dir / "results.json", orient="records", indent=2)
    save_plots(df, out_dir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
