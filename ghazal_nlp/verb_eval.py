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

from .verb import ZWNJ, Analyzer, Reading

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
