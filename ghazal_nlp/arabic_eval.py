"""The Arabic detector against my labels (data/arabic-*-lab.jsonl): whether each
span it found is Arabic, and whether lines I marked as holding Arabic get a span."""

from __future__ import annotations

import json
from collections import Counter

from . import arabic
from .orthography import DATA


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in open(DATA / name, encoding="utf-8")]


def evaluate(detector: arabic.ArabicDetector | None = None) -> dict:
    detector = detector or arabic.detector()
    spans = Counter()
    for r in rows("arabic-spans-lab.jsonl"):
        found = {(s.start, s.end) for s in detector.detect(r["text"])}
        spans[r["label"] if tuple(r["span"]) in found else "span_changed"] += 1
    judged = spans["arabic"] + spans["boundary"] + spans["not_arabic"]
    lines = Counter((r["label"], bool(detector.detect(r["text"]))) for r in rows("arabic-lines-lab.jsonl"))
    yes, no = lines["yes", True] + lines["yes", False], lines["no", True] + lines["no", False]
    return {"spans": dict(spans),
            "precision": round(spans["arabic"] / judged, 4) if judged else None,
            "precision_ignoring_boundaries": round((spans["arabic"] + spans["boundary"]) / judged, 4) if judged else None,
            "lines_with_arabic": yes, "found_in": lines["yes", True],
            "line_recall": round(lines["yes", True] / yes, 4) if yes else None,
            "lines_without_arabic": no, "false_alarms": lines["no", True]}
