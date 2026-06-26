from typing import Dict, List, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from connectors import build_connector


class ConnectorVLM(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg["models"]
        conn_cfg = cfg["connector"]
        self.image_processor = CLIPImageProcessor.from_pretrained(model_cfg["vision_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["lm_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vision = CLIPVisionModel.from_pretrained(model_cfg["vision_name"])
        self.lm = AutoModelForCausalLM.from_pretrained(model_cfg["lm_name"])
        if model_cfg.get("freeze_vision", True):
            self.vision.requires_grad_(False)
        if model_cfg.get("freeze_lm", True):
            self.lm.requires_grad_(False)

        clip_dim = int(self.vision.config.hidden_size)
        lm_dim = int(self.lm.config.hidden_size)
        self.connector = build_connector(conn_cfg["type"], clip_dim, lm_dim, conn_cfg)
        self.num_visual_tokens = int(conn_cfg["num_visual_tokens"])

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()
        self.lm.eval()
        return self

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.vision(pixel_values=pixel_values)
            return output.pooler_output

    def forward(self, pixel_values, input_ids, attention_mask, labels=None) -> Dict[str, torch.Tensor]:
        clip_features = self.encode_images(pixel_values)
        visual_embeds, aux = self.connector(clip_features)
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        visual_mask = torch.ones(
            attention_mask.size(0),
            self.num_visual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat([visual_mask, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            visual_labels = torch.full(
                (labels.size(0), self.num_visual_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([visual_labels, labels], dim=1)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)
        return {"loss": out.loss, "logits": out.logits, **aux}

    @torch.no_grad()
    def generate_answers(self, pixel_values, prompts: List[str], max_new_tokens: int = 12) -> List[str]:
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt").to(pixel_values.device)
        clip_features = self.encode_images(pixel_values)
        visual_embeds, _ = self.connector(clip_features)
        text_embeds = self.lm.get_input_embeddings()(tokenized["input_ids"])
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        visual_mask = torch.ones(
            tokenized["attention_mask"].size(0),
            self.num_visual_tokens,
            dtype=tokenized["attention_mask"].dtype,
            device=pixel_values.device,
        )
        attention_mask = torch.cat([visual_mask, tokenized["attention_mask"]], dim=1)
        generated = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def freeze_report(model: ConnectorVLM) -> Tuple[int, int, int]:
    vision = sum(p.numel() for p in model.vision.parameters() if p.requires_grad)
    lm = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
    connector = sum(p.numel() for p in model.connector.parameters() if p.requires_grad)
    return vision, lm, connector
