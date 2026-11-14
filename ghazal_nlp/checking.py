from bisect import bisect_left, bisect_right, insort
from itertools import accumulate
from dataclasses import asdict

from . import markdown, orthography


SEVERITY = {"violation": "error", "advisory": "advice", "house": "style"}
LEVEL = {"style": 0, "guess": 1, "advice": 2, "error": 3}


def check(text, *, format="text", style="ghazal", registers=("formal_written",),
          min_confidence="medium", severities=None, legacy=False):
    prose, at = markdown.view(text) if format == "markdown" else (text, None)
    result = orthography.rulebase(style).check(prose, registers=registers,
                                             min_confidence=min_confidence, severities=severities)
    findings, abstentions = [], []
    for f in result.findings:
        span = markdown.back(at, *f.span) if at is not None else f.span
        if span is not None:
            start, end = span
            finding = {"rule": f.rule_id, "start": start, "end": end, "text": text[start:end],
                       "suggestions": [] if f.suggestion is None else [f.suggestion],
                       "severity": SEVERITY[f.severity]}
            if legacy:
                finding.update(asdict(f), span=span, text=text[start:end])
            findings.append(finding)
    for f in result.abstentions:
        span = markdown.back(at, *f.span) if at is not None else f.span
        if span is not None:
            abstentions.append({**asdict(f), "span": span, "text": text[slice(*span)]})
    return {"findings": findings, "abstentions": abstentions, "rules_run": result.rules_run,
            "registers": list(result.registers), "rulebase_version": result.rulebase_version}


def merged(spans, findings):
    """Folds a round's edits into the earlier ones. A span is [start, end] in the text now, [start, end] in the text
    sent, the finding, and whether a later edit was merged into it."""
    by_end = sorted(spans, key=lambda s: s[1])
    ends, moved = [s[1] for s in by_end], [0, *accumulate(s[1] - s[0] - s[3] + s[2] for s in by_end)]
    sent = lambda p: p - moved[bisect_right(ends, p)]
    grow = lambda f: len(f["suggestions"][0]) - f["end"] + f["start"]
    items = sorted(([*s, [], (0, i)] for i, s in enumerate(spans)), key=lambda s: s[:2])
    for n, f in enumerate(findings):
        a, b = f["start"], f["end"]
        j = k = bisect_left(items, b, key=lambda s: s[0])
        while k and a < items[k - 1][1]:
            k -= 1
        hit = sorted(items[k:j], key=lambda s: s[7])
        lo, hi = min([a, *(s[0] for s in hit)]), max([b, *(s[1] for s in hit)])
        items[k:j] = []
        insort(items, [lo, hi, sent(lo), sent(hi), hit[0][4] if hit else f, bool(hit), [f, *(g for h in hit for g in h[6])], (1, n)], key=lambda s: s[:2])
    done = sorted((f["end"], grow(f)) for f in findings)
    done_ends, before = [e for e, _ in done], [0, *accumulate(g for _, g in done)]
    out = []
    for s in sorted(items, key=lambda s: (s[0], s[7])):
        shift = before[bisect_right(done_ends, s[0])] - sum(grow(f) for f in s[6] if f["end"] <= s[0])
        out.append([s[0] + shift, s[1] + shift + sum(map(grow, s[6])), *s[2:6]])
    return out



def fix(text, **options):
    sent, spans, seen = text, [], set()
    while text not in seen:
        seen.add(text)
        result = check(text, **options)
        edits, end = [], -1
        for f in result["findings"]:
            if (f["severity"] == "error" and len(f["suggestions"]) == 1
                    and f["start"] >= end and f["text"] != f["suggestions"][0]):
                edits.append(f)
                end = f["end"]
        if not edits:
            applied = [{**s[4], "start": s[2], "end": s[3], "text": sent[s[2]:s[3]],
                        "suggestions": [text[s[0]:s[1]]]} for s in spans]
            return {"text": text, "applied": applied, "left": result["findings"]}
        spans = merged(spans, edits)
        for f in reversed(edits):
            text = text[:f["start"]] + f["suggestions"][0] + text[f["end"]:]
        if len(seen) >= 20:
            break
    raise ValueError("fixes did not settle; use check and review the suggestions")
