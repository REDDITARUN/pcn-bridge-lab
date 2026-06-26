import base64
import io
import json
import pickle
import random
import ast
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def decode_m3it_image(value: Any) -> Image.Image:
    if isinstance(value, list):
        value = value[0]
    raw = base64.b64decode(value)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_question(example: Dict[str, Any]) -> str:
    instruction = str(example.get("instruction") or "").strip()
    inputs = str(example.get("inputs") or "").strip()
    if inputs:
        return f"{instruction}\n{inputs}".strip()
    return instruction


def build_answer(example: Dict[str, Any]) -> str:
    outputs = example.get("outputs", "")
    if isinstance(outputs, list):
        outputs = outputs[0]
    return str(outputs).strip()


class M3ITShapesDataset(Dataset):
    def __init__(self, hf_split, max_samples: int | None = None):
        if max_samples is not None:
            hf_split = hf_split.select(range(min(max_samples, len(hf_split))))
        self.split = hf_split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.split[idx]
        return {
            "image": decode_m3it_image(ex["image_base64_str"]),
            "question": build_question(ex),
            "answer": build_answer(ex),
        }


class DirectM3ITShapesDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], max_samples: int | None = None):
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        return {
            "image": decode_m3it_image(ex["image_base64_str"]),
            "question": build_question(ex),
            "answer": build_answer(ex),
        }


def load_m3it_shapes(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    if cfg.get("direct_hub_loader", True):
        return load_m3it_shapes_direct(cfg)
    try:
        ds = load_dataset(
            cfg["name"],
            cfg["config"],
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        )
        return {
            "train": M3ITShapesDataset(ds["train"], cfg.get("max_train_samples")),
            "validation": M3ITShapesDataset(ds["validation"], cfg.get("max_eval_samples")),
            "test": M3ITShapesDataset(ds["test"], cfg.get("max_eval_samples")),
        }
    except FileNotFoundError:
        return load_m3it_shapes_direct(cfg)


def load_m3it_shapes_direct(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    repo_id = cfg["name"]
    instructions_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data/vqa/shapes/instructions.json")
    with open(instructions_path, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    split_files = {
        "train": "data/vqa/shapes/train.large.jsonl",
        "validation": "data/vqa/shapes/val.jsonl",
        "test": "data/vqa/shapes/test.jsonl",
    }
    max_by_split = {
        "train": cfg.get("max_train_samples"),
        "validation": cfg.get("max_eval_samples"),
        "test": cfg.get("max_eval_samples"),
    }
    return {
        split: DirectM3ITShapesDataset(
            read_shapes_jsonl(repo_id, filename, instructions, max_by_split[split]),
            max_samples=None,
        )
        for split, filename in split_files.items()
    }


def read_shapes_jsonl(repo_id: str, filename: str, instructions: List[str], max_samples: int | None) -> List[Dict[str, Any]]:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    rows = []
    rng = random.Random(1234)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            ret = "yes" if data["answer"] == "true" else "no"
            rows.append({
                "instruction": rng.choice(instructions),
                "inputs": f"{data['question']}?",
                "image_base64_str": [data["image_str"]],
                "outputs": f"The answer is {ret}.",
            })
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


class CLEVRDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        image = ex["image"].convert("RGB")
        return {"image": image, "question": str(ex["question"]).strip(), "answer": str(ex["answer"]).strip()}


def load_clevr(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    if cfg["name"] == "dpdl-benchmark/clevr":
        return load_dpdl_clevr(cfg)
    streaming = bool(cfg.get("streaming", True))
    kwargs = dict(
        path=cfg["name"],
        name=cfg.get("config", "default"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        streaming=streaming,
    )
    max_train = cfg.get("max_train_samples")
    max_eval = cfg.get("max_eval_samples")
    train_rows = load_clevr_split(kwargs, "train", max_train or 5000, streaming, cfg.get("cache_dir"))
    val_rows = load_clevr_split(kwargs, "validation", max_eval or 500, streaming, cfg.get("cache_dir"))
    return {"train": CLEVRDataset(train_rows), "validation": CLEVRDataset(val_rows), "test": CLEVRDataset(val_rows)}


def decode_byte_string(value: Any) -> str:
    text = str(value)
    if text.startswith("b'") or text.startswith('b"'):
        try:
            return ast.literal_eval(text).decode("utf-8")
        except Exception:
            return text[2:-1]
    return text


def load_dpdl_clevr(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    max_train = int(cfg.get("max_train_samples") or 5000)
    max_eval = int(cfg.get("max_eval_samples") or 500)
    cache_dir = cfg.get("cache_dir")
    train_rows = load_dpdl_clevr_split("train", max_train, cache_dir)
    eval_rows = load_dpdl_clevr_split("test", max_eval, cache_dir)
    return {"train": CLEVRDataset(train_rows), "validation": CLEVRDataset(eval_rows), "test": CLEVRDataset(eval_rows)}


def load_dpdl_clevr_split(split: str, limit: int, cache_dir: str | None) -> List[Dict[str, Any]]:
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"dpdl_{split}_{limit}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)
    ds = load_dataset("dpdl-benchmark/clevr", split=split)
    rows = []
    for ex in tqdm(ds, total=len(ds), desc=f"flattening DPDL CLEVR {split} -> {limit} QA", leave=True):
        image = ex["image"].convert("RGB")
        qa = ex["question_answer"]
        for question, answer in zip(qa["question"], qa["answer"]):
            rows.append({"image": image, "question": decode_byte_string(question), "answer": decode_byte_string(answer)})
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(rows, f)
    return rows


def load_clevr_split(kwargs: Dict[str, Any], split: str, limit: int, streaming: bool, cache_dir: str | None) -> List[Dict[str, Any]]:
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{split}_{limit}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)
    ds = load_dataset(split=split, **kwargs)
    rows = []
    iterator = ds.take(limit) if streaming else (ds[i] for i in range(min(limit, len(ds))))
    for ex in tqdm(iterator, total=limit, desc=f"loading CLEVR {split}[:{limit}]", leave=True):
        rows.append({
            "image": ex["image"].convert("RGB"),
            "question": str(ex["question"]).strip(),
            "answer": str(ex["answer"]).strip(),
        })
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(rows, f)
    return rows


def load_vqa_dataset(cfg: Dict[str, Any]) -> Dict[str, Dataset]:
    kind = cfg.get("kind", "m3it_shapes")
    if kind == "clevr":
        return load_clevr(cfg)
    return load_m3it_shapes(cfg)


class VQACollator:
    def __init__(self, image_processor, tokenizer, max_length: int = 256):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [x["image"] for x in batch]
        questions = [x["question"] for x in batch]
        answers = [x["answer"] for x in batch]
        prompts = [format_prompt(q, self.tokenizer) for q in questions]
        full_texts = [p + a for p, a in zip(prompts, answers)]

        pixel_values = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.full_like(tokenized["input_ids"], -100)
        for i in range(labels.size(0)):
            seq_len = int(tokenized["attention_mask"][i].sum().item())
            answer_len = len(self.tokenizer(answers[i], add_special_tokens=False)["input_ids"])
            answer_len = max(1, min(answer_len, seq_len))
            start = seq_len - answer_len
            labels[i, start:seq_len] = tokenized["input_ids"][i, start:seq_len]

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
        }


def format_prompt(question: str, tokenizer=None) -> str:
    instruction = "Answer the visual question with exactly one word or one number. Do not explain."
    content = f"Question: {question}"
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{instruction}\n{content}\nAnswer: "
