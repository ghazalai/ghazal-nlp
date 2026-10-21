"""The spelling rules against PerSpellData's word-boundary errors: sentence
pairs that differ only in where a space or ZWNJ goes, or doesn't.

Only findings whose suggestion changes nothing but separators are scored. Each
gap a finding changes is one prediction; a gap where the wrong side and the
correct side differ is one gold edit. ``exact`` needs the same separator as
the correct side, ``apart`` only whether the two letters are kept apart (space
or ZWNJ) or joined, since the correct side often writes a space where the
house writes ZWNJ (می روم، شده اند).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .orthography import RuleBase

SEPARATORS = " ‌"
NAMES = {"": "none", " ": "space", "‌": "zwnj"}


def bare(text: str) -> str:
    return "".join(c for c in text if c not in SEPARATORS)


def gaps(text: str) -> list[str]:
    """The separator before each letter."""
    out, cur = [], ""
    for c in text:
        if c in SEPARATORS:
            cur += c
        else:
            out.append(cur)
            cur = ""
    return out


def pairs(wrong: Path, correct: Path):
    """Distinct (wrong, correct) lines that differ only in separators."""
    seen = set()
    with open(wrong, encoding="utf-8") as fw, open(correct, encoding="utf-8") as fc:
        for w, c in zip(fw, fc):
            w, c = w.strip(), c.strip()
            if w != c and bare(w) == bare(c) and (w, c) not in seen:
                seen.add((w, c))
                yield w, c


def changes(text: str, base: RuleBase, registers) -> dict[int, tuple[str, str]]:
    """{letter index: (proposed separator, severity)} from the separator-only findings."""
    out = {}
    for f in base.check(text, registers=registers).findings:
        a, b = f.span
        if f.suggestion is None or bare(f.suggestion) != bare(text[a:b]) or not bare(text[a:b]):
            continue
        first = len(bare(text[:a]))
        for k, (old, new) in enumerate(zip(gaps(text[a:b]), gaps(f.suggestion))):
            if k and old != new:
                out.setdefault(first + k, (new, f.severity))
    return out


def evaluate(files, base: RuleBase, registers=("formal_written",), limit: int | None = None) -> dict:
    report = {}
    for wrong, correct in files:
        t, by_kind, by_severity = Counter(), Counter(), Counter()
        for n, (w, c) in enumerate(pairs(wrong, correct)):
            if limit and n == limit:
                break
            gw, gc = gaps(w), gaps(c)
            t["pairs"] += 1
            t["gaps"] += len(gw) - 1
            edits = {i for i in range(1, len(gw)) if gw[i] != gc[i]}
            t["edits"] += len(edits)
            mine = changes(w, base, registers)
            t["changes_on_correct"] += len(changes(c, base, registers))
            for i in edits:
                kind = f"{NAMES.get(gw[i], 'mixed')}>{NAMES.get(gc[i], 'mixed')}"
                by_kind[kind, "edits"] += 1
                if i in mine and mine[i][0] == gc[i]:
                    by_kind[kind, "exact"] += 1
                if i in mine and bool(mine[i][0]) == bool(gc[i]):
                    by_kind[kind, "apart"] += 1
            for i, (sep, severity) in mine.items():
                t["changes"] += 1
                by_severity[severity, "changes"] += 1
                for name, ok in (("exact", sep == gc[i]), ("apart", bool(sep) == bool(gc[i]))):
                    t[name] += ok
                    by_severity[severity, name] += ok
                    t[f"{name}_edit"] += ok and i in edits
            t["sentences_fixed"] += mine.keys() == edits and all(mine[i][0] == gc[i] for i in edits)
        ratio = lambda a, b: round(a / b, 4) if b else 0.0
        report[Path(wrong).name] = {
            "pairs": t["pairs"], "edits": t["edits"], "edit_rate": ratio(t["edits"], t["gaps"]),
            "changes": t["changes"],
            "precision": {"exact": ratio(t["exact"], t["changes"]), "apart": ratio(t["apart"], t["changes"])},
            "recall": {"exact": ratio(t["exact_edit"], t["edits"]), "apart": ratio(t["apart_edit"], t["edits"])},
            "sentences_fixed": ratio(t["sentences_fixed"], t["pairs"]),
            "changes_on_correct": t["changes_on_correct"],
            "by_severity": {s: {"changes": by_severity[s, "changes"],
                                "exact": ratio(by_severity[s, "exact"], by_severity[s, "changes"]),
                                "apart": ratio(by_severity[s, "apart"], by_severity[s, "changes"])}
                            for s in sorted({s for s, _ in by_severity})},
            "recall_by_edit": {k: {"edits": by_kind[k, "edits"], "exact": ratio(by_kind[k, "exact"], by_kind[k, "edits"]),
                                   "apart": ratio(by_kind[k, "apart"], by_kind[k, "edits"])}
                               for k in sorted({k for k, _ in by_kind}, key=lambda k: -by_kind[k, "edits"])},
        }
    return report
