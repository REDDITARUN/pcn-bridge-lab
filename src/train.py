import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VQACollator, load_vqa_dataset
from eval import evaluate
from model import ConnectorVLM, freeze_report
from optimizers import EqPropMomentum, build_optimizer
from utils import append_jsonl, count_trainable_parameters, get_peak_memory_gb, load_config, make_run_dir, save_json, set_seed


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_one_epoch(model, loader, optimizer, device, cfg, epoch, metrics_path):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.time()
    aux_weight = float(cfg["connector"].get("aux_energy_weight", 0.0))
    precision = cfg["training"].get("mixed_precision", "bf16")
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    log_every = int(cfg["training"].get("log_every", 25))

    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for step, batch in enumerate(pbar, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if isinstance(optimizer, EqPropMomentum):
            if grad_accum != 1:
                raise ValueError("EqPropMomentum path currently expects grad_accum_steps=1.")
            with autocast_context(device, precision):
                with torch.no_grad():
                    free = model(pixel_values, input_ids, attention_mask, labels=None)
                clamped = model(pixel_values, input_ids, attention_mask, labels)
                beta = float(cfg.get("eqprop", {}).get("beta", 0.1))
                free_energy = free.get("energy", clamped["loss"].new_zeros(())).detach()
                clamped_energy = clamped.get("energy", clamped["loss"].new_zeros(())) + beta * clamped["loss"]
                settle_penalty = clamped.get("settle_delta")
            loss_value = float(clamped["loss"].detach().item())
            physics_loss = optimizer.step(free_energy=free_energy, clamped_energy=clamped_energy, settle_penalty=settle_penalty)
            optimizer.zero_grad(set_to_none=True)
        else:
            with autocast_context(device, precision):
                out = model(pixel_values, input_ids, attention_mask, labels)
                loss = out["loss"] + aux_weight * out.get("energy", out["loss"].new_zeros(()))
                loss = loss / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.connector.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.item() * grad_accum)
            physics_loss = None
        losses.append(loss_value)
        pbar.set_postfix(loss=f"{loss_value:.4f}")
        if step % log_every == 0:
            row = {
                "phase": "train_step",
                "epoch": epoch,
                "step": step,
                "loss": loss_value,
                "peak_memory_gb": get_peak_memory_gb(),
            }
            if physics_loss is not None:
                row["physics_loss"] = physics_loss
            append_jsonl(metrics_path, row)
    return {"loss": sum(losses) / max(len(losses), 1), "seconds": time.time() - started}


def save_checkpoint(model, optimizer, run_dir: Path, epoch: int, cfg):
    path = run_dir / f"checkpoint-epoch{epoch}.pt"
    torch.save({
        "connector": model.connector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg,
    }, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m3it_shapes_smol.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(int(cfg["seed"]))
    conn_type = cfg["connector"]["type"]
    run_name = args.name or f"{conn_type}-s{cfg['connector'].get('settle_steps', 0)}-{cfg['training']['optimizer']}"
    run_dir = make_run_dir(cfg["output_dir"], run_name)
    metrics_path = run_dir / "metrics.jsonl"
    save_json(run_dir / "config.json", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = ConnectorVLM(cfg).to(device)
    vision_trainable, lm_trainable, connector_trainable = freeze_report(model)
    if vision_trainable != 0 or lm_trainable != 0:
        raise RuntimeError("Frozen backbone check failed: CLIP and SmolLM2 must have zero trainable params.")

    ds = load_vqa_dataset(cfg["dataset"])
    collator = VQACollator(model.image_processor, model.tokenizer)
    train_loader = DataLoader(
        ds["train"],
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    optimizer = build_optimizer(cfg["training"]["optimizer"], model.connector.parameters(), cfg["training"], cfg.get("eqprop"))
    append_jsonl(metrics_path, {
        "phase": "setup",
        "connector_params": connector_trainable,
        "total_trainable_params": count_trainable_parameters(model),
        "device": str(device),
    })

    best_val = float("inf")
    best_path = None
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch, metrics_path)
        append_jsonl(metrics_path, {"phase": "train_epoch", "epoch": epoch, **train_metrics, "peak_memory_gb": get_peak_memory_gb()})
        if epoch % int(cfg["training"].get("eval_every_epochs", 1)) == 0:
            val_metrics = evaluate(model, val_loader, device, cfg["generation"]["max_new_tokens"], generate=False)
            append_jsonl(metrics_path, {"phase": "validation", "epoch": epoch, **val_metrics, "peak_memory_gb": get_peak_memory_gb()})
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_path = save_checkpoint(model, optimizer, run_dir, epoch, cfg)
        elif epoch % int(cfg["training"].get("save_every_epochs", 1)) == 0:
            save_checkpoint(model, optimizer, run_dir, epoch, cfg)

    save_json(run_dir / "summary.json", {"best_validation_loss": best_val, "best_checkpoint": str(best_path) if best_path else None})
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
