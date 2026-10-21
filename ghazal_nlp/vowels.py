"""مصوت‌گذاری: the vowels Persian editors write on each consonant skeleton.

Tokens are canonicalised mechanically, merging only encodings of one spelling.
The marks on a token's last letter are syntax (the ezāfe, an Arabic case
ending, a pausal sukun) and are split off. The remaining marked forms of a
skeleton group into readings: two forms are two readings only when they write
different vowels on the same letter, since an unmarked letter says nothing. A
skeleton with one reading is settled; one nobody marked is absent, not settled.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"
SNAPSHOT = "Ganjoor API snapshot api-2026-08-03, every poem's text with the editors' marks (ganjoor.poems)"

ZWNJ, ZWJ, TATWEEL = "\u200c", "\u200d", "\u0640"
FATHATAN, DAMMATAN, KASRATAN = "\u064b", "\u064c", "\u064d"
FATHA, DAMMA, KASRA, SHADDA, SUKUN = "\u064e", "\u064f", "\u0650", "\u0651", "\u0652"
HAMZA, HAMZA_BELOW, HEH = "\u0654", "\u0655", "ه"
VOWELS = frozenset(FATHATAN + DAMMATAN + KASRATAN + FATHA + DAMMA + KASRA + SUKUN + "\u0670")
HARAKAT = VOWELS | {SHADDA}

MARKS = "\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u08d3-\u08ff"
TOKEN = re.compile(rf"(?:[^\W_]|[{MARKS}{ZWNJ}{ZWJ}{TATWEEL}])+")
PERSIAN = re.compile("[ؠ-يٮ-ۿ]")
MARKS_AFTER_CONTROL = re.compile(rf"([{ZWNJ}{ZWJ}]+)([{MARKS}]+)")

# ۀ is written هٔ, and repaired where NFD already turned its ه into Kurdish ە;
# then decomposed hamza carriers are composed.
REWRITES = (
    ("\u06c0", HEH + HAMZA), ("\u06d5" + HAMZA, HEH + HAMZA), ("ا" + HAMZA, "أ"), ("ا" + HAMZA_BELOW, "إ"),
    ("و" + HAMZA, "ؤ"), ("ي" + HAMZA, "ئ"), ("ی" + HAMZA, "ئ"),
)
FOLD = str.maketrans("يىۍكڪ", "یییکک")

TERMINAL = {KASRA: "ezafe", HAMZA: "ezafe_he", DAMMA: "i3rab", FATHA: "i3rab", DAMMATAN: "i3rab",
            KASRATAN: "i3rab", FATHATAN: "i3rab", SUKUN: "pausal"}

# The folds a search index makes and a lexicon refuses, each priced by the build.
REFUSED_FOLDS = {
    "teh_marbuta": {"ة": HEH},
    "hamza_carriers": {"أ": "ا", "إ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ی"},
    "alef_maddah": {"آ": "ا"},
    "zwnj": {ZWNJ: "", ZWJ: ""},
    "heh_hamza": {HAMZA: ""},
}


def tokens(text: str) -> list[str]:
    """Word tokens, with their marks, ZWNJ and tatweel inside them."""
    return TOKEN.findall(text)


def is_persian(token: str) -> bool:
    return bool(PERSIAN.search(token))


def slots(form: str) -> list[list[str]]:
    """[letter, marks] pairs, each mark once; marks before the first letter modify nothing."""
    out = []
    for ch in form:
        if not unicodedata.combining(ch):
            out.append([ch, ""])
        elif out and ch not in out[-1][1]:
            out[-1][1] += ch
    return out


def _render(parsed) -> str:
    return "".join(letter + "".join(sorted(marks, key=unicodedata.combining)) for letter, marks in parsed)


def skeleton(form: str) -> str:
    return "".join(ch for ch in form if not unicodedata.combining(ch))


def is_marked(form: str) -> bool:
    return any(ch in HARAKAT for ch in form)


def canonicalise(token: str) -> str:
    """One encoding per spelling: tatweel dropped, marks typed after a ZWNJ moved
    onto their letter, ۀ and hamza carriers as above, Arabic yeh and kaf folded,
    marks in Unicode canonical order and each once, edge ZWNJs dropped. Sukun, ة,
    hamza carriers, آ and ZWNJ are kept: folding them can merge two words. Never
    empties a token."""
    out = token.replace(TATWEEL, "") or token
    out = MARKS_AFTER_CONTROL.sub(r"\2\1", out)
    for old, new in REWRITES:
        out = out.replace(old, new)
    out = _render(slots(out.translate(FOLD)))
    return out.strip(ZWNJ + ZWJ) or out or token


def split_terminal(form: str) -> tuple[str, str]:
    """(lexical form, marks on the last letter). Persian has no word-final short
    vowel, so those are syntax. A one-letter word keeps its mark (وَ, وُ), as do
    a shadda, a hamza not on ه, and the tanvin of اصلاً."""
    parsed = slots(form)
    if len(parsed) < 2:
        return form, ""
    letter, marks = parsed[-1]
    terminal = "".join(m for m in sorted(marks, key=unicodedata.combining) if m in TERMINAL
                       and (m != HAMZA or letter == HEH) and (m != FATHATAN or letter != "ا"))
    if not terminal:
        return form, ""
    parsed[-1][1] = "".join(m for m in marks if m not in terminal)
    return _render(parsed), terminal


@dataclass(frozen=True, slots=True)
class Reading:
    form: str
    count: int
    variants: int


def _vowelled(form: str) -> int:
    return sum(1 for _, marks in slots(form) if VOWELS.intersection(marks))


def readings_of(forms: dict[str, int]) -> tuple[Reading, ...]:
    """Marked forms of one skeleton grouped into readings.

    Forms writing different vowels on one letter never share a reading. The most
    vowelled forms come first and found the readings; a partial form joins one
    that already carries all its marks, else the first it does not contradict.
    A reading is shown as its most vowelled attested spelling.
    """
    groups = []
    for form, count in sorted(forms.items(), key=lambda kv: (-_vowelled(kv[0]), -len(kv[0]), -kv[1], kv[0])):
        parsed = [set(marks) for _, marks in slots(form)]
        fits = [g for g in groups if all(not (a & VOWELS and b & VOWELS) or a & VOWELS == b & VOWELS
                                         for a, b in zip(parsed, g[0]))]
        chosen = next((g for g in fits if all(a <= b for a, b in zip(parsed, g[0]))), fits[0] if fits else None)
        if chosen is None:
            groups.append((parsed, [(form, count)]))
            continue
        for a, b in zip(parsed, chosen[0]):
            b |= a
        chosen[1].append((form, count))
    out = [Reading(max(members, key=lambda fc: (_vowelled(fc[0]), fc[1], fc[0]))[0], sum(c for _, c in members),
                   len(members)) for _, members in groups]
    return tuple(sorted(out, key=lambda r: (-r.count, r.form)))


@dataclass(frozen=True, slots=True)
class Entry:
    skeleton: str
    readings: tuple[Reading, ...]
    marked: int
    total: int
    ezafe: int

    @property
    def settled(self) -> bool:
        return len(self.readings) == 1

    def ambiguous(self, min_evidence: int = 1) -> bool:
        """Two readings attested ``min_evidence`` times or more each."""
        return sum(r.count >= min_evidence for r in self.readings) > 1


class Lexicon(dict):
    """Skeleton to Entry, for every skeleton an editor marked."""

    def lookup(self, token: str) -> Entry | None:
        return self.get(skeleton(split_terminal(canonicalise(token))[0]))

    def settled(self) -> list[Entry]:
        return [e for e in self.values() if e.settled]

    def ambiguous(self, min_evidence: int = 1) -> list[Entry]:
        return [e for e in self.values() if e.ambiguous(min_evidence)]

    @classmethod
    def load(cls, path: Path = DATA / "vowels-lexicon.tsv") -> Lexicon:
        readings, totals = {}, {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                key, form, count, variants, *rest = line.split("\t")
                readings.setdefault(key, []).append(Reading(form, int(count), int(variants)))
                totals[key] = tuple(map(int, rest))
        return cls({key: Entry(key, tuple(r), *totals[key]) for key, r in readings.items()})


@lru_cache(maxsize=1)
def lexicon() -> Lexicon:
    return Lexicon.load()


def attested(counts: dict[str, int]):
    """Per skeleton: canonical marked forms with counts, and the totals of all,
    marked and ezāfe-carrying attestations."""
    forms, total, marked, ezafe = {}, Counter(), Counter(), Counter()
    for token, count in counts.items():
        if not is_persian(token):
            continue
        lexical, terminal = split_terminal(canonicalise(token))
        if not (key := skeleton(lexical)):
            continue
        total[key] += count
        if any(TERMINAL[m].startswith("ezafe") for m in terminal):
            ezafe[key] += count
        if is_marked(lexical):
            forms.setdefault(key, Counter())[lexical] += count
            marked[key] += count
    return forms, total, marked, ezafe


def build_lexicon(counts: dict[str, int]) -> Lexicon:
    forms, total, marked, ezafe = attested(counts)
    return Lexicon({k: Entry(k, readings_of(f), marked[k], total[k], ezafe[k]) for k, f in forms.items()})


def measure(counts: dict[str, int], lex: Lexicon, min_evidence: int = 1) -> dict:
    """Running tokens whose skeleton is ambiguous, settled, or never marked."""
    ambiguous_keys = {e.skeleton for e in lex.ambiguous(min_evidence)}
    n = Counter()
    for token, count in counts.items():
        if is_persian(token) and (key := skeleton(split_terminal(canonicalise(token))[0])):
            n["tokens"] += count
            n["ambiguous" if key in ambiguous_keys else "settled" if key in lex else "unattested"] += count
    total, decided = n["tokens"], n["ambiguous"] + n["settled"]
    return {"tokens": total, "ambiguous": n["ambiguous"], "settled": n["settled"], "unattested": n["unattested"],
            **{f"{k}_share": round(n[k] / total if total else 0.0, 6) for k in ("ambiguous", "settled", "unattested")},
            "ambiguous_share_of_decided": round(n["ambiguous"] / decided if decided else 0.0, 6)}


def mark_census(counts: dict[str, int]) -> dict[str, int]:
    """Where the marks go: lexical, the ezāfe, or the other final-letter classes."""
    census = Counter()
    for token, count in counts.items():
        canonical = canonicalise(token)
        census["harakat_raw"] += sum(ch in HARAKAT for ch in token) * count
        census["harakat_canonical"] += sum(ch in HARAKAT for ch in canonical) * count
        lexical, terminal = split_terminal(canonical)
        for m in terminal:
            census[TERMINAL[m]] += count
            if m in HARAKAT:
                census["terminal_harakat"] += count
        census["lexical_harakat"] += sum(ch in HARAKAT for ch in lexical) * count
        census["heh_hamza"] += canonical.count(HEH + HAMZA) * count
    return dict(census)


def refusal_census(counts: dict[str, int], lex: Lexicon) -> dict[str, int]:
    """How many distinct words each refused merge would have collapsed."""
    forms = attested(counts)[0]
    initial_shadda = [c for f in forms.values() for form, c in f.items() if SHADDA in slots(form)[0][1]]
    sukun = 0
    for entry in lex.ambiguous():
        without = Counter()
        for form, count in forms[entry.skeleton].items():
            without[form.replace(SUKUN, "")] += count
        sukun += len(readings_of(without)) == 1
    census = {"initial_shadda_forms": len(initial_shadda), "initial_shadda_tokens": sum(initial_shadda),
              "sukun_deletion_would_merge_skeletons": sukun, "vowel_conflict_skeletons": len(lex.ambiguous())}
    for name, fold in REFUSED_FOLDS.items():
        table, merged = str.maketrans(fold), Counter()
        for key in lex:
            merged[key.translate(table)] += 1
        census[f"{name}_would_merge_skeletons"] = sum(n for n in merged.values() if n > 1)
    return census


def review_queue(lex: Lexicon) -> list[tuple[str, str, int, tuple[Reading, ...]]]:
    """(skeleton, reason, minority count, readings) for every unsettled skeleton."""
    items = []
    for e in lex.values():
        if e.settled:
            continue
        minority = e.readings[1].count
        parsed = [slots(r.form) for r in e.readings]
        if any(len(VOWELS.intersection(marks)) > 1 for p in parsed for _, marks in p):
            reason = "malformed_attestation"
        elif any(p and SHADDA in p[0][1] for p in parsed):
            reason = "impossible_initial_shadda"
        else:
            reason = "hapax_reading" if minority == 1 else "rare_reading" if minority < 5 else "attested_homograph"
        items.append((e.skeleton, reason, minority, e.readings))
    return sorted(items, key=lambda item: (-item[2], item[0]))


def derive(gold: Path):
    """The lexicon, review queue and report from ``ganjoor.poems`` written as JSON lines."""
    counts = Counter()
    with open(gold, encoding="utf-8") as f:
        for line in f:
            counts.update(tokens(json.loads(line)["text"]))
    before = {}
    for token in counts:
        if is_marked(token):
            before.setdefault(skeleton(token), set()).add(token)
    lex = build_lexicon(counts)
    queue = review_queue(lex)
    report = {
        "skeletons_multiply_marked_before": sum(len(f) > 1 for f in before.values()),
        "skeletons_with_a_reading": len(lex),
        "skeletons_settled": len(lex.settled()),
        "skeletons_ambiguous": len(lex.ambiguous()),
        "ambiguous_by_min_evidence": {n: len(lex.ambiguous(n)) for n in (1, 2, 3, 5, 10)},
        "review_queue": dict(Counter(reason for _, reason, _, _ in queue)),
        "refused": refusal_census(counts, lex),
        "marks": mark_census(counts),
        "running_text": measure(counts, lex),
        "ambiguous_share_by_min_evidence": {n: measure(counts, lex, n)["ambiguous_share"] for n in (1, 2, 3, 5, 10)},
    }
    return lex, queue, report


def write(lex: Lexicon, queue, report: dict, data: Path = DATA) -> None:
    provenance = [f"# source: {SNAPSHOT}", "# split: all poems; nothing here is scored on held-out text",
                  f"# derived: {date.today().isoformat()}"]
    header = [
        "# Persian vowel readings: each consonant skeleton and the vocalizations editors",
        "# wrote on it, after canonicalising the marks and removing those on the last letter.",
        "# reading: the most vowelled spelling of a group of forms that never write different",
        "# vowels on one letter. count: its attestations; variants: spellings merged into it;",
        "# marked, total, ezafe: the skeleton's marked, all and ezāfe-carrying attestations.",
        "# A skeleton nobody marked is absent: unknown, not unambiguous.",
        *provenance,
        *(f"# {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in report.items()),
        "# skeleton\treading\tcount\tvariants\tmarked\ttotal\tezafe",
    ]
    rows = [f"{e.skeleton}\t{r.form}\t{r.count}\t{r.variants}\t{e.marked}\t{e.total}\t{e.ezafe}"
            for e in sorted(lex.values(), key=lambda e: e.skeleton) for r in e.readings]
    (data / "vowels-lexicon.tsv").write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    header = [
        "# Skeletons with more than one reading, the most attested second reading first.",
        "# malformed_attestation: a letter carries two vowels. impossible_initial_shadda: a",
        "# reading has shadda on the first letter. hapax_reading, rare_reading: the second",
        "# reading is attested once, or two to four times. attested_homograph: five or more.",
        *provenance,
        "# skeleton\treason\tminority\treadings",
    ]
    rows = [f"{k}\t{reason}\t{minority}\t" + " / ".join(f"{r.form}={r.count}" for r in readings)
            for k, reason, minority, readings in queue]
    (data / "vowels-review.tsv").write_text("\n".join(header + rows) + "\n", encoding="utf-8")
