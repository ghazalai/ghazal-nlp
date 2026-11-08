"""The opt-in ezāfe model: one TookaBERT fine-tuned for both decisions, the unwritten ezāfe (``junction``) and
the ezāfe written as ه (``heh``). It needs torch and transformers (``pip install ghazal-nlp[neural]``) and the
model's folder."""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import ezafe


def _marked(text: str, head: tuple[int, int], nxt: tuple[int, int], width: int = 120) -> str:
    (h0, h1), (n0, n1) = head, nxt
    while n0 < n1 and text[n0].isspace():
        n0 += 1
    return text[max(0, h0 - width):h0] + "«" + text[h0:h1] + " " + text[n0:n1] + "»" + text[n1:n1 + width]


class Ezafe:
    def __init__(self, path: str | Path, device: str | None = None, batch: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.batch = torch, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.stack = json.loads((Path(path) / "heh-stack.json").read_text(encoding="utf-8"))

    def logits(self, task: str, items: list[tuple[str, tuple[int, int], tuple[int, int]]]) -> list[float]:
        """The model's log-odds for each (text, head span, next span); task ``J`` or ``H``."""
        texts = [task + " " + _marked(*item) for item in items]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tokenizer(texts[i:i + self.batch], truncation=True, max_length=96, padding=True,
                                     return_tensors="pt").to(self.device)
                out += self.model(**enc).logits[:, 0].float().tolist()
        return out

    def junctions(self, text: str, profile: str = "prose", metre: str | None = None) -> list[ezafe.Junction]:
        """``ezafe.junctions``, with the model where the trees would decide."""
        found = [s for s in ezafe.sites({"text": text, "metre": metre})
                 if not s["marked"] and ezafe.junction(s, profile, lambda _: 0.0).reason == "model"]
        z = self.logits("J", [(s["neutralized_text"], s["neutral_head_span"], s["neutral_next_span"]) for s in found])
        p = {tuple(s["head_span"]): 1 / (1 + math.exp(-v)) for s, v in zip(found, z)}
        return ezafe.junctions(text, profile, metre, lambda s: p[tuple(s["head_span"])])

    def heh(self, text: str, threshold: float = 0.9) -> list[ezafe.Heh]:
        """``ezafe.heh``, with the model's log-odds corrected by trees over the same features."""
        return ezafe.heh(text, threshold, self.heh_probabilities)

    def heh_probabilities(self, found: list[dict]) -> list[float]:
        z = self.logits("H", [(s["window"], s["head_span"], s["next_span"]) for s in found])
        return [ezafe._sigmoid(v + ezafe._boosted(self.stack, s["features"] | {"neural": v})) for s, v in zip(found, z)]
