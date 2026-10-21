"""ezafe and vowels against sentences with full pronunciations: SentenceBench
(400 sentences, ezāfe written -e or -ye) and HomoRich's test split (127,850
sentences on 285 homographs, ezāfe written e1). HomoRich's symbols are mapped
to SentenceBench's (/ → a, a → A, $ → S, c → C, @ → ?).

Words are paired inside a sentence by their consonants. A vowel reading fits
when the pronunciation has the vowels it marks; letters it leaves unmarked fit
anything. The vowel lexicon is Ganjoor's, so this is verse readings on modern
prose.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from . import ezafe, vowels

WORD = re.compile(r"[ء-ۓً-ْٰٔ‌]+")
MARKS = re.compile("[ً-ْٰٔ]")
CONSONANTS = dict(zip("بپتطثسصجچحهخدذزضظرژشغقفکگلمن", "bpttsssjChhxdzzzzrZSqqfkglmn"))
SHORT = {"َ": "[aA]", "ِ": "[ei]", "ُ": "[ou]", "ْ": ""}
HOMORICH = str.maketrans({"/": "a", "a": "A", "$": "S", "c": "C", "@": "?"})
EZAFE = re.compile(r"-y?e$")


def rows(path: Path):
    """(sentence, pronunciation words, homograph, its pronunciation)."""
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if "grapheme" in r:
                yield r["grapheme"], r["phoneme"].split(), r["homograph word"], r["pronunciation"]
            else:
                words = [re.sub(r"(y?e)1$", r"-\1", w).translate(HOMORICH) for w in r["Mapped Phoneme"].split()]
                yield r["Grapheme"], words, r["Homograph Grapheme"], r["Mapped Homograph Phoneme"].translate(HOMORICH)


def bare_sound(word: str) -> str:
    return EZAFE.sub("", word).replace("-", "")


def skel_sound(word: str) -> str:
    s = re.sub(r"(.)\1", r"\1", re.sub("[aeoiuAvy?.,!؟،-]", "", bare_sound(word)))
    return s[:-1] if s.endswith("h") else s


def skel_letters(word: str) -> str:
    w = re.sub("[هۀ]‌?ی?$", "", MARKS.sub("", word))
    return "".join(CONSONANTS.get(c, "") for c in w)


def words(sentence: str, sounds: list[str]):
    fw = [(w, skel_letters(w)) for w in WORD.findall(sentence)]
    pw = [(w, skel_sound(w)) for w in sounds]
    fw, pw = [x for x in fw if x[1]], [x for x in pw if x[1]]
    blocks = SequenceMatcher(None, [k for _, k in fw], [k for _, k in pw], autojunk=False).get_matching_blocks()
    return [(fw[a + i][0], pw[b + i][0]) for a, b, n in blocks for i in range(n)]


def pattern(form: str) -> re.Pattern:
    slots, out = vowels.slots(form), []
    for i, (ch, marks) in enumerate(slots):
        vowel = next((SHORT[m] for m in marks if m in SHORT), None)
        if ch == "ا":
            base, vowel = ("\\??", vowel if vowel is not None else "[aeoA]?") if i == 0 else ("A?", vowel or "")
        elif ch == "آ":
            base = "\\??A"
        elif ch == "و":
            base = "v" if i == 0 else "(?:v|u|o|ow)?"
        elif ch == "ی":
            base = "y" if i == 0 else "(?:y|i|ey)?"
        elif ch == "ه" and i == len(slots) - 1:
            base = "(?:e|h|)"
        elif ch in "عءئأؤ":
            base = "\\??"
        elif ch in CONSONANTS:
            base = CONSONANTS[ch]
        else:
            return re.compile("(?!)")
        if "ّ" in marks:
            base += base + "?"
        out.append(base + ("[aeo]?" if vowel is None else vowel))
    return re.compile("".join(out))


def fits(form: str, sound: str) -> bool:
    return bool(pattern(form).fullmatch(bare_sound(sound)))


def evaluate(path: Path, seed: int = 0) -> dict:
    t, confusion = Counter(), {"written": Counter(), "junction": Counter()}
    lexicon, rng = vowels.lexicon(), random.Random(seed)
    for sentence, sounds, homograph, said in rows(path):
        pairs = words(sentence, sounds)
        t["sentences"] += 1
        t["words"] += len(WORD.findall(sentence))
        t["paired"] += len(pairs)
        t["ezafe"] += sum(bool(EZAFE.search(s)) for _, s in pairs)
        for word, sound in pairs:
            entry = lexicon.lookup(word)
            if not entry or not fits(vowels.skeleton(MARKS.sub("", word)), sound):
                continue
            ok = [fits(r.form, sound) for r in entry.readings]
            for kind in ("settled" if entry.settled else "ambiguous",) + (("homograph",) if MARKS.sub("", word) == homograph else ()):
                t[kind] += 1
                t[f"{kind}_top"] += ok[0]
                t[f"{kind}_any"] += any(ok)
                t[f"{kind}_random"] += ok[rng.randrange(len(ok))]
        for site in ezafe.sites({"text": sentence}):
            head = WORD.match(sentence, site["head_span"][0])
            sound = next((s for w, s in pairs if head and w == head[0]), None)
            if sound is None:
                continue
            gold = "ezafe" if EZAFE.search(sound) or sound.endswith("eye") else "none"
            confusion["junction"][f"{gold}->{ezafe.junction(site).label}"] += 1
            if site["marked"]:
                d = ezafe.written(site).label
                confusion["written"][f"{gold}->{'none' if d == 'not_ezafe' else d}"] += 1
    rate = lambda a, b: round(t[a] / t[b], 4) if t[b] else 0.0
    report = {"sentences": t["sentences"], "words_paired": rate("paired", "words"), "ezafe_rate": rate("ezafe", "paired"),
              "vowels": {k: {"words": t[k], "top_reading_fits": rate(f"{k}_top", k), "some_reading_fits": rate(f"{k}_any", k),
                             "random_reading_fits": rate(f"{k}_random", k)} for k in ("settled", "ambiguous", "homograph")}}
    for which, c in confusion.items():
        n = sum(c.values())
        right = sum(v for k, v in c.items() if k.split("->")[0] == k.split("->")[1])
        gold = Counter(k.split("->")[0] for k in c.elements())
        report[f"ezafe_{which}"] = {"sites": n, "agree": round(right / n, 4) if n else 0.0,
                                    "majority_baseline": round(max(gold.values()) / n, 4) if n else 0.0,
                                    "confusion": dict(sorted(c.items()))}
    return report
