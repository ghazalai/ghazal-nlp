"""vowels and ezafe against the Tajik Shahnameh (data/shahnameh-tajik-test.jsonl).

Tajik Cyrillic writes the short vowels and the ezāfe, so I aligned the Tajik
copy to Ganjoor's hemistich by hemistich and put its vowels on the Persian
letters. The rows are 3,000 hemistichs from the held-out fifth of the poems
(aruz_eval.held_out), each word with its vocalized form and whether an ezāfe
follows. Only words the alignment accounted for letter by letter are kept.

The vowels are the Tajik edition's (یَکی, سَر, و as u), and Tajik doesn't
tell i from ī or u from ū, so a long vowel without a letter reads as short.
Tajik writes every word-final silent ه as -a, so the letter before one fits any
vowel. A reading fits when every vowel it marks is the Tajik one on that
letter, every sukun it marks has none, and a vowel it marks before a bare ا، و
or ی is that letter's (چُو، رَا). The lexicon comes from all of Ganjoor, these
poems included, but from the editors' marks, not from the Tajik.
"""

from __future__ import annotations

import json
import random
from collections import Counter

from . import ezafe, vowels
from .orthography import DATA

SHORT = {vowels.FATHA: "a", vowels.KASRA: "i", vowels.DAMMA: "u"}
NATURAL = {"ا": "a", "آ": "a", "و": "u", "ی": "i"}
FOLD = str.maketrans({"أ": "ا", "إ": "ا", "ؤ": "و", "ي": "ی", "ى": "ی", "ك": "ک", "\u200c": None, "\u200d": None})


def rows(path=DATA / "shahnameh-tajik-test.jsonl") -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def letters(word: str) -> str:
    return vowels.skeleton(word).translate(FOLD)


def marks(form: str) -> list[str | None]:
    """Per letter: a, i, u, "" for sukun, None for unmarked."""
    out = []
    for ch, m in vowels.slots(form.replace("\u200c", "")):
        out.append(next((SHORT[c] for c in m if c in SHORT), "" if vowels.SUKUN in m else None))
    return out


def fits(reading: str, gold: str) -> bool:
    r, g, chars = marks(reading), marks(gold), [ch for ch, _ in vowels.slots(gold.replace("\u200c", ""))]
    if chars[-1] == "ه" and len(g) > 1 and g[-2] is None:
        g[-2] = "?"
    after = [NATURAL.get(ch) for ch in chars[1:]] + [None]
    return len(r) == len(g) and all(a is None or b == "?" or a == (b or "") or (b is None and n == a and m is None)
                                    for a, b, n, m in zip(r, g, after, g[1:] + ["-"]))


def heads(text: str, words: list) -> dict[int, list]:
    """Each kept word by the offset of its token in ``text``: the kept words are an
    ordered subsequence of the tokens."""
    out, i = {}, 0
    for t in ezafe.WORD.finditer(text):
        if i < len(words) and letters(t.group()) == letters(words[i][0]):
            out[t.start()] = words[i]
            i += 1
    return out


def evaluate(path=DATA / "shahnameh-tajik-test.jsonl", seed: int = 0) -> dict:
    t, sites, lexicon, rng = Counter(), [], vowels.lexicon(), random.Random(seed)
    for r in rows(path):
        t["hemistichs"] += 1
        for word, voc, _ in r["words"]:
            t["words"] += 1
            entry = lexicon.lookup(word)
            if not entry or len(marks(entry.readings[0].form)) != len(marks(voc)):
                continue
            ok = [fits(x.form, voc) for x in entry.readings]
            kind = "settled" if entry.settled else "ambiguous"
            t[kind] += 1
            t[f"{kind}_top"] += ok[0]
            t[f"{kind}_any"] += any(ok)
            t[f"{kind}_random"] += ok[rng.randrange(len(ok))]
        found = heads(r["fa"], r["words"])
        for site in ezafe.sites({"text": r["fa"]}):
            if (w := found.get(site["head_span"][0])) is not None:
                sites.append(site | {"gold": "ezafe" if w[2] else "none", "gold_written": "ezafe" if w[2] else "not_ezafe"})
    rate = lambda a, b: round(t[a] / t[b], 4) if t[b] else 0.0
    return {"hemistichs": t["hemistichs"], "words": t["words"],
            "vowels": {k: {"words": t[k], "top_reading_fits": rate(f"{k}_top", k), "some_reading_fits": rate(f"{k}_any", k),
                           "random_reading_fits": rate(f"{k}_random", k)} for k in ("settled", "ambiguous")},
            "ezafe_junction": ezafe.score(sites, ezafe.junction, "gold"),
            "ezafe_written": ezafe.score([s for s in sites if s["marked"]], ezafe.written, "gold_written")}
