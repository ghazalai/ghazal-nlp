"""The scanner next to other metre classifiers, on the same Ganjoor-Rhythm-Bench
couplets: the ones from held-out poems. A model's predictions come as JSON
lines (``poem_id``, ``couplet_index``, ``metre`` in feet, ``known``: the gold
metre is one the model has, ``in_training``: a hemistich of the couplet is in
the model's training data). Every rate is exact metre per couplet, and the
majority baseline always answers the commonest gold metre of that subset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .aruz import Scanner, scan_poem
from .aruz_eval import held_out, tidy


def couplets(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if held_out(str(r["poem_id"])):
                out.setdefault((r["poem_id"], r["couplet_index"]), r)
    return out


def scanner_metres(rows: dict, scanner: Scanner) -> dict:
    feet = {label: tidy(label).split(" (")[0] for label in scanner.inventory.label_patterns}
    return {k: {feet.get(m, m) for m in scan_poem(r["first"] + "\n" + r["second"], scanner).metres}
            for k, r in rows.items()}


def compare(bench: Path, predictions: list[Path], scanner: Scanner) -> dict:
    rows = couplets(bench)
    gold = {k: tidy(r["rhythm"]) for k, r in rows.items()}
    mine = scanner_metres(rows, scanner)
    report = {"couplets": len(rows)}
    for path in predictions:
        with open(path, encoding="utf-8") as f:
            model = {(p["poem_id"], p["couplet_index"]): p for p in map(json.loads, f)}
        subsets = {"all": list(model), "known": [k for k, p in model.items() if p["known"]],
                   "known_unseen": [k for k, p in model.items() if p["known"] and not p["in_training"]]}
        report[Path(path).stem] = {
            name: {"couplets": len(keys),
                   "model": round(sum(model[k]["metre"] == gold[k] for k in keys) / len(keys), 4) if keys else 0.0,
                   "scanner": round(sum(mine[k] == {gold[k]} for k in keys) / len(keys), 4) if keys else 0.0,
                   "majority": round(Counter(gold[k] for k in keys).most_common(1)[0][1] / len(keys), 4) if keys else 0.0}
            for name, keys in subsets.items()}
    return report
