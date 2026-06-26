import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VQACollator, load_vqa_dataset
from eval_binary import candidate_losses
from model import ConnectorVLM
from utils import parse_value, save_json, set_nested, set_seed


CLEVR_ANSWERS = [
    "yes", "no",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
    "cube", "sphere", "cylinder",
    "small", "large",
    "rubber", "metal",
]


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().replace(".", "").split())


@torch.no_grad()
def evaluate_choice(model, loader, device, candidates):
    model.eval()
    correct = 0
    total = 0
    chosen_losses = []
    pred_counts = {c: 0 for c in candidates}
    for batch in tqdm(loader, desc="choice-eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        losses = []
        for candidate in candidates:
            losses.append(candidate_losses(model, pixel_values, batch["prompts"], candidate))
        loss_mat = torch.stack(losses, dim=1)
        pred_idx = loss_mat.argmin(dim=1).detach().cpu().tolist()
        min_loss = loss_mat.min(dim=1).values.detach().cpu().tolist()
        chosen_losses.extend(min_loss)
        for idx, answer in zip(pred_idx, batch["answers"]):
            pred = candidates[idx]
            pred_counts[pred] += 1
            correct += int(normalize(pred) == normalize(answer))
            total += 1
    return {
        "choice_accuracy": correct / max(total, 1),
        "choice_loss": sum(chosen_losses) / max(len(chosen_losses), 1),
        "n": total,
        "prediction_counts": pred_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    for item in args.override:
        key, value = item.split("=", 1)
        set_nested(cfg, key.split("."), parse_value(value))
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConnectorVLM(cfg).to(device)
    model.connector.load_state_dict(ckpt["connector"])
    ds = load_vqa_dataset(cfg["dataset"])
    loader = DataLoader(
        ds[args.split],
        batch_size=int(cfg["training"]["batch_size"]),
        collate_fn=VQACollator(model.image_processor, model.tokenizer),
        num_workers=0,
    )
    metrics = evaluate_choice(model, loader, device, CLEVR_ANSWERS)
    out_path = Path(args.checkpoint).with_suffix(f".{args.split}.choice_metrics.json")
    save_json(out_path, metrics)
    print({k: v for k, v in metrics.items() if k != "prediction_counts"})


if __name__ == "__main__":
    main()
