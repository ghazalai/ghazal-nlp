"""Word language in context, learned from silver labels; stdlib inference."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache

from .arabic import DATA, GAP, MARKS, TOKEN, ArabicSpan, detector, fold_word


@dataclass(frozen=True)
class ArabicWord:
    start: int
    end: int
    probability: float
    arabic: bool
    kind: str | None = None
    quran_refs: tuple[str, ...] = ()
    verified: tuple[int, int] | None = None


def grams(word):
    word = '^' + word + '$'
    return [word[i:i + 3] for i in range(len(word) - 2)]


def boundary(text):
    return bool(GAP.search(text)) or any(c.isalnum() for c in text)


class WordModel:
    def __init__(self, data):
        self.data = data
        self.threshold = data.get('threshold', 0.5)

    @lru_cache(maxsize=32768)
    def lexical(self, raw):
        word = fold_word(raw)
        scores = [self.data['grams'].get(g, 0) for g in grams(word)]
        return [self.data['words'].get(word, 0), sum(scores) / max(1, len(scores)),
                min(2, len(MARKS.findall(raw)) / max(1, len(word))),
                float(any(c in word for c in 'پچژگ')), float('ة' in word),
                float('‌' in raw), min(12, len(word)) / 6]

    def features(self, text, *, strict=False, rules=None):
        tokens = list(TOKEN.finditer(text))
        breaks = [True] + [boundary(text[a.end():b.start()])
                           for a, b in zip(tokens, tokens[1:])] + [True]
        rules = detector().detect(text) if rules is None else rules
        rule = [0] * len(tokens)
        j = 0
        for i, t in enumerate(tokens):
            while j < len(rules) and rules[j].end <= t.start():
                j += 1
            if j < len(rules) and rules[j].start <= t.start() < rules[j].end:
                rule[i] = 1 if rules[j].kind == 'phrase' else 2
        base = [self.lexical(t[0]) for t in tokens] if not strict or any(rule) else []
        feats = []
        for i, t in enumerate(tokens):
            if strict and not rule[i]:
                feats.append(None)
                continue
            row = list(base[i])
            for offset in (-2, -1, 1, 2):
                k = i + offset
                inside = 0 <= k < len(tokens) and not any(breaks[min(i, k) + 1:max(i, k) + 1])
                row.extend(base[k][:3] if inside else [0, 0, 0])
            a, b = i, i + 1
            while a > max(0, i - 4) and not breaks[a]:
                a -= 1
            while b < min(len(tokens), i + 5) and not breaks[b]:
                b += 1
            row.extend(sum(x[k] for x in base[a:b]) / (b - a) for k in range(4))
            row.extend([float(breaks[i]), float(breaks[i + 1]), rule[i],
                        float('«' in text[max(0, t.start() - 2):t.start()]),
                        float('»' in text[t.end():t.end() + 2])])
            feats.append(row)
        return tokens, feats, breaks

    def probability(self, row):
        score = self.data.get('bias', 0)
        for tree in self.data['trees']:
            node = 0
            while tree[node][0] >= 0:
                feature, threshold, left, right = tree[node]
                node = left if row[feature] <= threshold else right
            score += tree[node][1]
        score = max(-40, min(40, score / self.data.get('temperature', 1)))
        calibration = self.data.get('calibration', [1, 0])
        score = max(-40, min(40, calibration[0] * score + calibration[1]))
        return 1 / (1 + math.exp(-score))

    def words(self, text, *, quran=None, threshold=None, strict=False):
        exact = quran.detect(text) if quran is not None else []
        return self._words(text, exact, threshold, strict)

    def _words(self, text, exact, threshold, strict, *, spans_only=False, rules=None):
        if threshold is None:
            threshold = self.data.get('strict_threshold', .995) if strict else self.threshold
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0, 1]')
        tokens, feats, breaks = self.features(text, strict=spans_only and strict, rules=rules)
        out, j = [], 0
        for t, row in zip(tokens, feats):
            while j < len(exact) and exact[j].end <= t.start():
                j += 1
            match = exact[j] if j < len(exact) and exact[j].start <= t.start() < exact[j].end else None
            p = self.probability(row) if row is not None else 0.0
            out.append(ArabicWord(t.start(), t.end(), 1.0 if match else p,
                                  bool(match) or (row is not None and p >= threshold and (not strict or row[25] > 0)),
                                  'quran' if match else None,
                                  match.quran_refs if match else (),
                                  (match.start, match.end) if match else None))
        a = 0
        for b in range(1, len(out) + 1):
            if b < len(out) and not breaks[b] and out[b].arabic == out[a].arabic and out[b].verified == out[a].verified:
                continue
            if b - a == 1 and out[a].arabic and out[a].verified is None:
                out[a] = replace(out[a], arabic=False)
            a = b
        return out

    def detect(self, text, *, quran=None, verse=False, threshold=None, strict=False):
        exact = quran.detect(text) if quran is not None else []
        rules = detector().detect(text) if strict else None
        if strict and not rules:
            cutoff = self.data.get('strict_threshold', .995) if threshold is None else threshold
            if not 0 <= cutoff <= 1:
                raise ValueError('threshold must lie in [0, 1]')
            return exact
        words = self._words(text, exact, threshold, strict, spans_only=True, rules=rules)
        found, run = list(exact), []

        def emit():
            if not run:
                return
            a, b = run[0].start, run[-1].end
            key = ' '.join(fold_word(t[0]) for t in TOKEN.finditer(text[a:b]))
            kind = 'formula' if key in detector().formulae else 'verse' if verse else 'passage'
            found.append(ArabicSpan(a, b, ('word_model',), kind=kind))
            run.clear()

        for word in words:
            if run and (not word.arabic or word.verified or boundary(text[run[-1].end:word.start])):
                emit()
            if word.arabic and not word.verified:
                run.append(word)
        emit()
        return sorted(found, key=lambda s: s.start)


@lru_cache(maxsize=1)
def model():
    with gzip.open(DATA / 'arabic-model.json.gz', 'rt', encoding='utf-8') as f:
        return WordModel(json.load(f))
