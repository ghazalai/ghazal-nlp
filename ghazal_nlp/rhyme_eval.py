"""The rhyme finder against Ganjoor's rhyme letters, on the poems ``ganjoor``
writes out. Ganjoor gives each poem's rhyme letters with the radif and no
spaces; a masnavi has none, so its couplets are only counted."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import rhyme
from .aruz_eval import held_out


def comparable(letters: str) -> str:
    """Ganjoor writes an ezāfe هٔ as هی in some poems and as ه in others."""
    return letters.replace("هی", "ه")


def evaluate(path: Path, split: str = "held_out", tolerance: float = rhyme.TOLERANCE, misses: int = 0) -> dict:
    tally, examples = Counter(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("form") not in rhyme.FORMS or held_out(p["id"]) != (split == "held_out"):
                continue
            form, lines = p["form"], p["text"].split("\n")
            found = rhyme.scan(lines, form, tolerance)
            if form == "masnavi":
                tally["masnavi_couplets"] += len(found)
                tally["masnavi_rhymed"] += sum(r.abstention is None and not r.off for r in found)
                continue
            gold = rhyme.key(p.get("rhyme_letters") or "")
            if not gold:
                continue
            r = found[0]
            tally[f"{form}:poems"] += 1
            if r.abstention:
                tally[f"{form}:abstained"] += 1
            elif comparable(r.letters) == comparable(gold):
                tally[f"{form}:agree"] += 1
            elif len(examples) < misses:
                examples.append({"id": p["id"], "ganjoor": p["rhyme_letters"], "mine": r.letters, "radif": r.radif,
                                 "words": r.words, "off": r.off})
    report = {"split": split, "tolerance": tolerance}
    for form in rhyme.FORMS:
        n = tally[f"{form}:poems"]
        if n:
            report[form] = {"poems": n, "agree": round(tally[f"{form}:agree"] / n, 4),
                            "abstained": round(tally[f"{form}:abstained"] / n, 4)}
    if tally["masnavi_couplets"]:
        report["masnavi"] = {"couplets": tally["masnavi_couplets"],
                             "rhymed": round(tally["masnavi_rhymed"] / tally["masnavi_couplets"], 4)}
    if examples:
        report["misses"] = examples
    return report
