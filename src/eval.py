import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VQACollator, load_vqa_dataset
from model import ConnectorVLM
from utils import exact_match, load_config, parse_value, safe_mean, save_json, set_nested, set_seed


@torch.no_grad()
def evaluate(model, loader, device, max_new_tokens=12, generate=False):
    model.eval()
    losses, accs = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(pixel_values, input_ids, attention_mask, labels)
        losses.append(float(out["loss"].item()))
        if generate:
            preds = model.generate_answers(pixel_values, batch["prompts"], max_new_tokens=max_new_tokens)
            accs.extend(exact_match(p, a) for p, a in zip(preds, batch["answers"]))
    return {"loss": safe_mean(losses), "accuracy": safe_mean(accs) if generate else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m3it_shapes_smol.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config") or load_config(args.config)
    for item in args.override:
        key, value = item.split("=", 1)
        set_nested(cfg, key.split("."), parse_value(value))
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConnectorVLM(cfg).to(device)
    model.connector.load_state_dict(ckpt["connector"])
    ds = load_vqa_dataset(cfg["dataset"])
    collator = VQACollator(model.image_processor, model.tokenizer)
    loader = DataLoader(ds[args.split], batch_size=cfg["training"]["batch_size"], collate_fn=collator)
    metrics = evaluate(model, loader, device, cfg["generation"]["max_new_tokens"], args.generate)
    out_path = Path(args.checkpoint).with_suffix(f".{args.split}.metrics.json")
    save_json(out_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
