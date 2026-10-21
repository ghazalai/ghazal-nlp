"""The verb analyser against UD treebank verbs (CoNLL-U).

Lemma and features are scored apart, because the treebanks' conventions differ
from each other and from the analyser's: Seraji lemmatises passive شد as کرد,
the future auxiliary خواهد carries Tense=Fut, PerDT does not mark participles,
and negation is not always marked. A reading agrees on features when it matches
every one of Person, Number, Polarity and Tense that both sides give, Tense
except Fut (the future is two tokens here), and when gold says VerbForm=Part it
is a participle or a perfect. Mood is scored separately, on the verbs the
treebank gives one; none of the three marks Aspect, so it isn't scored.
"""

from __future__ import annotations

from collections import Counter

from .verb import PAST_ENDINGS, PRESENT_ENDINGS, ZWNJ, Analyzer, Reading

FEATURES = ("Person", "Number", "Polarity", "Tense")
MOODS = {"Indicative": "Ind", "Subjunctive": "Sub", "Imperative": "Imp"}


def verbs(path):
    """(form, lemma, feats) for each VERB or AUX word, and every word's form."""
    out, words = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) != 10 or not row[0].isdigit():
                continue
            words.append(row[1])
            if row[3] in ("VERB", "AUX"):
                feats = dict(kv.split("=", 1) for kv in row[5].split("|") if "=" in kv)
                out.append((row[1], row[2].split("#")[0], feats))
    return out, words


def agrees(reading: Reading, gold: dict, mood: bool = False) -> bool:
    mine = dict(kv.split("=") for kv in reading.features().split("|"))
    if gold.get("VerbForm") == "Part" and reading.verb_form != "participle" and reading.aspect not in (
            "perfect", "imperfective_perfect"):
        return False
    if mood and MOODS[mine["Mood"]] != gold["Mood"]:
        return False
    return all(mine[k] == gold[k] for k in FEATURES if k in mine and k in gold and gold[k] != "Fut")


def evaluate(paths, analyzer: Analyzer) -> dict:
    tally, reasons, unknown = Counter(), Counter(), Counter()
    for path in paths:
        rows, words = verbs(path)
        tally["words"] += len(words)
        mi = [w for w in words if w.startswith(("می" + ZWNJ, "نمی" + ZWNJ))]
        tally["mi_words"] += len(mi)
        tally["mi_analysed"] += sum(bool(analyzer.analyze(w).readings) for w in mi)
        for form, lemma, gold in rows:
            tally["verbs"] += 1
            a = analyzer.analyze(form)
            if not a.readings:
                reasons[a.abstention] += 1
                unknown[form] += 1
                continue
            tally["analysed"] += 1
            tally["readings"] += len(a.readings)
            hits = sum(agrees(r, gold) for r in a.readings)
            tally["lemma_found"] += any(r.lemma[:-1] == lemma for r in a.readings)
            tally["agreeing_found"] += hits > 0
            if "Mood" in gold:
                tally["mood_scored"] += 1
                tally["mood_found"] += any(agrees(r, gold, mood=True) for r in a.readings)
            if len(a.readings) == 1:
                tally["sole"] += 1
                tally["sole_agrees"] += hits
    n = tally["verbs"] or 1
    return {
        "verbs": tally["verbs"],
        "analysed": round(tally["analysed"] / n, 4),
        "lemma_among_readings": round(tally["lemma_found"] / n, 4),
        "features_among_readings": round(tally["agreeing_found"] / n, 4),
        "features_when_analysed": round(tally["agreeing_found"] / (tally["analysed"] or 1), 4),
        "mean_readings": round(tally["readings"] / (tally["analysed"] or 1), 2),
        "sole_reading": round(tally["sole"] / (tally["analysed"] or 1), 4),
        "sole_reading_features": round(tally["sole_agrees"] / (tally["sole"] or 1), 4),
        "mood_scored": tally["mood_scored"],
        "features_and_mood_when_analysed": round(tally["mood_found"] / (tally["mood_scored"] or 1), 4),
        "mi_words": tally["mi_words"],
        "mi_analysed": round(tally["mi_analysed"] / (tally["mi_words"] or 1), 4),
        "abstentions": dict(reasons.most_common()),
        "commonest_unanalysed": unknown.most_common(20),
    }


# UniMorph cells conjugate can build, as its arguments; the others (perfects, future, progressive, the bare
# present PFV, participles) are counted, not scored.
CELLS = {("PRS", "IPFV"): {}, ("PST",): {"tense": "past"}, ("PST", "IPFV"): {"tense": "past", "aspect": "imperfective"},
         ("SBJV", "PRS"): {"mood": "subjunctive"}, ("IMP",): {"mood": "imperative"}}


def naive(infinitive: str, tense="present", mood="indicative", person=3, number="sing", aspect=None, **_) -> str:
    """The baseline: every verb regular, the present stem the infinitive less یدن or its last two letters."""
    stem = infinitive[:-1] if tense == "past" else infinitive[:-3] if infinitive.endswith("یدن") else infinitive[:-2]
    endings = PAST_ENDINGS if tense == "past" else PRESENT_ENDINGS
    ending = next(e for e, v in endings.items() if v == (person, number))
    if mood == "imperative":
        ending = "" if number == "sing" else "ید"
    prefix = "می" if mood == "indicative" and (tense == "present" or aspect) else "ب" if tense == "present" else ""
    return prefix + stem + ending


def unimorph(path, analyzer: Analyzer) -> dict:
    """conjugate against UniMorph's Persian paradigms, cell by cell. UniMorph
    writes no ZWNJ (میخورم), so forms are compared without it. A compound verb
    (آبتنی کردن) is its light verb's form after the first part."""
    tally, misses = Counter(), []
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.count("\t") == 2]
    for lemma, form, tags in rows:
        t = tags.split(";")
        if "COL" in t:
            continue
        person = next((int(x) for x in t if x in "123"), None)
        cell = next((c for c in CELLS if set(c) == set(t) - {"V", "1", "2", "3", "SG", "PL"}), None)
        kind = "compound" if " " in lemma else "simple"
        if cell is None or person is None:
            tally[f"{kind}:not_built"] += 1
            continue
        first, _, verb = lemma.rpartition(" ")
        args = dict(CELLS[cell], person=person, number="sing" if "SG" in t else "plur")
        mine, base = analyzer.conjugate(verb, **args), naive(verb, **args)
        gold = form.replace(ZWNJ, "")
        tally[f"{kind}:cells"] += 1
        tally[f"{kind}:built"] += mine is not None
        tally[f"{kind}:right"] += mine is not None and (first + " " if first else "") + mine.replace(ZWNJ, "") == gold
        tally[f"{kind}:baseline"] += (first + " " if first else "") + base == gold
        if mine is not None and (first + " " if first else "") + mine.replace(ZWNJ, "") != gold and len(misses) < 30:
            misses.append([lemma, tags, form, mine])
    report = {}
    for kind in ("simple", "compound"):
        n = tally[f"{kind}:cells"] or 1
        report[kind] = {"cells": tally[f"{kind}:cells"], "not_built_cells": tally[f"{kind}:not_built"],
                        "built": round(tally[f"{kind}:built"] / n, 4), "right": round(tally[f"{kind}:right"] / n, 4),
                        "right_when_built": round(tally[f"{kind}:right"] / (tally[f"{kind}:built"] or 1), 4),
                        "naive_baseline": round(tally[f"{kind}:baseline"] / n, 4)}
    report["misses"] = misses
    return report
