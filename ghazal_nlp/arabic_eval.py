"""The Arabic detector against my labels (data/arabic-*-lab.jsonl): whether each
span it found is Arabic, and whether lines I marked as holding Arabic get a span."""

from __future__ import annotations

import json
from collections import Counter
from time import perf_counter

from . import arabic
from .orthography import DATA


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in open(DATA / name, encoding="utf-8")]


def evaluate(detector: arabic.ArabicDetector | None = None) -> dict:
    detector = detector or arabic.detector()
    spans = Counter()
    labels, predictions = set(), set()
    for r in rows("arabic-spans-lab.jsonl"):
        found = {(s.start, s.end) for s in detector.detect(r["text"])}
        labels.add((r['text'], *r['span']))
        predictions.update((r['text'], a, b) for a, b in found)
        spans[r["label"] if tuple(r["span"]) in found else "span_changed"] += 1
    judged = spans["arabic"] + spans["boundary"] + spans["not_arabic"]
    lines = Counter((r["label"], bool(detector.detect(r["text"]))) for r in rows("arabic-lines-lab.jsonl"))
    yes, no = lines["yes", True] + lines["yes", False], lines["no", True] + lines["no", False]
    return {"spans": dict(spans),
            "precision_scope": "unchanged previously judged spans only; not precision of new predictions",
            "unjudged_predicted_spans": len(predictions - labels),
            "precision": round(spans["arabic"] / judged, 4) if judged else None,
            "precision_ignoring_boundaries": round((spans["arabic"] + spans["boundary"]) / judged, 4) if judged else None,
            "lines_with_arabic": yes, "found_in": lines["yes", True],
            "line_recall": round(lines["yes", True] / yes, 4) if yes else None,
            "lines_without_arabic": no, "false_alarms": lines["no", True]}


def annotated(records, predict=arabic.spans) -> dict:
    """Complete span annotations: text, spans [[start,end,kind]], category.

    Exact boundaries and word overlap are measured separately. Kind recall
    measures detection of that gold kind, regardless of predicted kind.
    The old lab span judgments are incomplete and cannot be passed here.
    """
    groups, kinds, typed, confusions = {}, {}, {}, Counter()
    elapsed, total = 0.0, 0
    for row in records:
        text, gold = row['text'], row['spans']
        weight = row.get('weight', 1)
        if weight <= 0:
            raise ValueError('weight must be positive')
        start = perf_counter()
        found = predict(text)
        elapsed += perf_counter() - start
        tokens = list(arabic.TOKEN.finditer(text))
        total += len(tokens)
        counts = Counter()
        for token in tokens:
            actual = next((s[2] for s in gold if s[0] <= token.start() < s[1]), None)
            guess = any(s.start <= token.start() < s.end for s in found)
            counts['tp' if actual and guess else 'fn' if actual else 'fp' if guess else 'tn'] += 1
            if actual:
                kinds.setdefault(actual, Counter())['tp' if guess else 'fn'] += weight
            if guess and not actual:
                confusions[arabic.fold_word(token[0])] += weight
        expected = {(s[0], s[1]) for s in gold}
        predicted = {(s.start, s.end) for s in found}
        for kind in {s[2] for s in gold} | {s.kind for s in found}:
            actual_kind = {(s[0], s[1]) for s in gold if s[2] == kind}
            found_kind = {(s.start, s.end) for s in found if s.kind == kind}
            typed.setdefault(kind, Counter()).update(tp=weight * len(actual_kind & found_kind),
                                                     fp=weight * len(found_kind - actual_kind),
                                                     fn=weight * len(actual_kind - found_kind))
        counts.update(span_tp=len(expected & predicted), span_fp=len(predicted - expected),
                      span_fn=len(expected - predicted), lines=1, positive_lines=bool(gold),
                      found_lines=bool(gold) and counts['tp'] > 0,
                      false_alarm_lines=not gold and bool(found), negative_lines=not gold)
        for key in dict.fromkeys(('all', row.get('category', 'unknown'))):
            groups.setdefault(key, Counter()).update({k: v * weight for k, v in counts.items()})

    def metrics(c):
        result = dict(c)
        for prefix in ('', 'span_'):
            tp, fp, fn = (c[prefix + k] for k in ('tp', 'fp', 'fn'))
            result[prefix + 'precision'] = tp / (tp + fp) if tp + fp else None
            result[prefix + 'recall'] = tp / (tp + fn) if tp + fn else None
        result['line_recall'] = c['found_lines'] / c['positive_lines'] if c['positive_lines'] else None
        result['false_alarms_per_1000_lines'] = 1000 * c['false_alarm_lines'] / c['lines']
        return result

    return {'categories': {k: metrics(v) for k, v in groups.items()},
            'typed_spans': {k: {'tp': v['tp'], 'fp': v['fp'], 'fn': v['fn'],
                               'precision': v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] else None,
                               'recall': v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] else None}
                            for k, v in typed.items()},
            'kind_detection': {k: {'tp': v['tp'], 'fn': v['fn'],
                                  'recall': v['tp'] / (v['tp'] + v['fn'])} for k, v in kinds.items()},
            'false_positive_words': confusions.most_common(30),
            'words': total, 'seconds': elapsed, 'words_per_second': total / elapsed if elapsed else None}
