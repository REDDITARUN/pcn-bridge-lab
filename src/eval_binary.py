import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VQACollator, load_vqa_dataset
from model import ConnectorVLM
from utils import parse_value, save_json, set_nested, set_seed


def target_label(answer: str) -> str:
    text = answer.lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    raise ValueError(f"Cannot parse binary answer: {answer}")


def make_candidate_batch(tokenizer, prompts, candidate, max_length=256):
    full_texts = [p + candidate for p in prompts]
    tokenized = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.full_like(tokenized["input_ids"], -100)
    for i in range(labels.size(0)):
        seq_len = int(tokenized["attention_mask"][i].sum().item())
        answer_len = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        answer_len = max(1, min(answer_len, seq_len))
        start = seq_len - answer_len
        labels[i, start:seq_len] = tokenized["input_ids"][i, start:seq_len]
    return tokenized["input_ids"], tokenized["attention_mask"], labels


@torch.no_grad()
def candidate_losses(model, pixel_values, prompts, candidate):
    input_ids, attention_mask, labels = make_candidate_batch(model.tokenizer, prompts, candidate)
    input_ids = input_ids.to(pixel_values.device)
    attention_mask = attention_mask.to(pixel_values.device)
    labels = labels.to(pixel_values.device)
    out = model(pixel_values, input_ids, attention_mask, labels=None)
    visual_labels = torch.full(
        (labels.size(0), model.num_visual_tokens),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    full_labels = torch.cat([visual_labels, labels], dim=1)
    shift_logits = out["logits"][:, :-1].contiguous()
    shift_labels = full_labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    return (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def evaluate_binary(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    for batch in tqdm(loader, desc="binary-eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        yes_loss = candidate_losses(model, pixel_values, batch["prompts"], "yes")
        no_loss = candidate_losses(model, pixel_values, batch["prompts"], "no")
        preds = torch.where(yes_loss < no_loss, 1, 0).tolist()
        for pred, answer in zip(preds, batch["answers"]):
            gold = 1 if target_label(answer) == "yes" else 0
            correct += int(pred == gold)
            total += 1
        losses.extend(torch.minimum(yes_loss, no_loss).detach().cpu().tolist())
    return {"binary_accuracy": correct / max(total, 1), "binary_loss": sum(losses) / max(len(losses), 1), "n": total}


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
    metrics = evaluate_binary(model, loader, device)
    out_path = Path(args.checkpoint).with_suffix(f".{args.split}.binary_metrics.json")
    save_json(out_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
