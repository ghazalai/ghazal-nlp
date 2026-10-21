"""The ezāfe after a silent ه: where the junctions are, and whether one is there.

``sites`` finds every word ending in a proven silent ه followed by another
word, records how the junction is written (هٔ، ۀ، ه‌ی، هی، ه ی، or nothing), and
returns the text with only that junction's mark removed. The two classifiers
read only that neutralized text: ``written`` decides whether a written mark is
an ezāfe, and ``junction`` whether an unmarked junction carries one.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from . import orthography, verb, vowels

ZWNJ, ZWJ = "‌", "‍"
WORD = re.compile(r"[\wً-ٰٟۖ-ۭ‌‍]+")
MARKS = "ً-ٰٟۖ-ۭ"
ENDING = re.compile(rf"(?P<heh>[هۀە])(?P<marks>[{MARKS}‌‍]*)(?P<yeh>[یيى])?(?P<last>[{MARKS}]*)$")

CONNECTORS = frozenset("را که و یا اما ولی اگر مگر زیرا بلکه گر ور".split())
COPULAS = frozenset("است هست نیست هستم هستی هستیم هستید هستند نیستم نیستی نیستیم نیستید نیستند "
                    "بود بودم بودی بودیم بودید بودند باشد باشم باشی باشیم باشید باشند".split())
PRONOUNS = frozenset("من تو او ما شما ایشان آنان آنها آن‌ها وی خود خودش خودم خودت خودمان خودتان خودشان "
                     "خویش همدیگر یکدیگر".split())
CLITICS = frozenset("ای ام ات اش ایم اید اند".split())
PREPOSITIONS = frozenset("از به با در بر برای بدون بی تا جز نزد پیش پس چون مانند همچون سوی زیر روی میان "
                         "اندر ز کز".split())


def inventory() -> dict[str, dict[str, int]]:
    rows = (orthography.DATA / "orthography-silent-he.tsv").read_text(encoding="utf-8").splitlines()
    return {w.replace(ZWNJ, ""): {"silent": int(s), "voiced": int(v), "bare": int(b)}
            for w, s, v, b in (line.split("\t") for line in rows if line and not line.startswith("#"))}


@lru_cache(maxsize=65536)
def bare(token: str) -> str:
    return vowels.skeleton(vowels.canonicalize(token)).replace(ZWJ, "")


def has_letter(word: str) -> bool:
    return any(unicodedata.category(c).startswith("L") for c in word)


@lru_cache(maxsize=32768)
def readings(token: str) -> verb.Analysis:
    return verb.analyzer().analyze(bare(token))


def sites(segment: dict, lexicon: dict | None = None):
    """Each silent-ه junction in ``segment["text"]``, with the segment's own fields.

    Offsets address the original string. Only the junction's hamza, kasra, ی and
    joiners are removed; other marks and the punctuation in the gap stay.
    """
    lexicon = inventory() if lexicon is None else lexicon
    text = segment["text"]
    words = list(WORD.finditer(text))
    for index, token in enumerate(words[:-1]):
        match = ENDING.search(token.group())
        if not match:
            continue
        glyph, marks, yeh = match["heh"], match["marks"] + match["last"], match["yeh"]
        if glyph == "ە" and "ٔ" not in marks:
            continue  # a real Kurdish ae
        raw_head = token.group()[:match.start()] + "ه"
        head = bare(raw_head).replace(ZWNJ, "")
        if head not in lexicon:
            continue
        following = words[index + 1]
        mark = ("zwnj_yeh" if ZWNJ in marks or ZWJ in marks else "solid_yeh") if yeh else \
            {"ۀ": "precomposed_heh", "ە": "nfd_damaged_heh"}.get(glyph) or \
            ("heh_hamza" if "ٔ" in marks else "heh_kasra" if "ِ" in marks else "unmarked")
        if (not yeh and bare(following.group()).replace(ZWNJ, "") == "ی"
                and text[token.end():following.start()].isspace() and index + 2 < len(words)
                and text[following.end():words[index + 2].start()].isspace()):
            mark = "spaced_yeh"
            following = words[index + 2]
        if not has_letter(following.group()):
            continue
        kept = "".join(c for c in marks if c not in "ِٔ" + ZWNJ + ZWJ)
        neutral_head = raw_head + kept
        suffix_end = words[index + 1].end() if mark == "spaced_yeh" else token.end()
        gap = text[suffix_end:following.start()]
        replacement = neutral_head + (" " if not gap.strip() else f" {gap.strip()} ")
        head_span, next_span = [token.start(), token.start() + match.start() + 1], [following.start(), following.end()]
        identity = [segment.get("source"), segment.get("segment_id"), text, head_span, next_span]
        start = token.start() + len(replacement)
        yield {
            **segment,
            "site_id": "ezafe-" + hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest(),
            "head": head, "head_span": head_span, "next_span": next_span,
            "written_head_span": [token.start(), suffix_end],
            "observed_mark_class": mark, "marked": mark != "unmarked",
            "original_junction": text[token.start():following.start()],
            "neutralized_text": text[:token.start()] + replacement + text[following.start():],
            "neutral_head_span": [token.start(), token.start() + len(neutral_head)],
            "neutral_next_span": [start, start + len(following.group())],
            "lexicon_evidence": lexicon[head],
        }


def _parts(site: dict):
    text = site["neutralized_text"]
    h0, h1 = site["neutral_head_span"]
    n0, n1 = site["neutral_next_span"]
    if not 0 <= h0 < h1 <= n0 <= n1 <= len(text):
        raise ValueError("invalid neutral spans")
    return text, text[h0:h1], text[h1:n0], text[n0:n1], n1


@dataclass(frozen=True, slots=True)
class Decision:
    label: str
    reason: str


# Tendencies, not laws: before چون، به، با، در a written mark is still an ezāfe
# 44–74% of the time, so these lists justify keeping a spelling, not asserting it.
WRITTEN_PREPOSITIONS = frozenset("از به با در بر تا ز کز اندر برای".split())
SIMILE = frozenset("چون همچون".split())
HARD_PUNCTUATION = frozenset("،,.;؛:!?؟…)]}»”")


def written(site: dict, arm: str = "closed") -> Decision:
    """Is the written هٔ، ۀ or ه‌ی at ``site`` an ezāfe? Yes, unless the next thing
    is the end, hard punctuation, a connector, a simple preposition, a simile word
    or a copula (or, in arm ``closed+verb``, a finite verb). The other readings are
    the old spelling of ‌ای: the indefinite or the second-person copula."""
    if arm not in ("closed", "closed+verb"):
        raise ValueError(f"unknown arm {arm!r}")
    _, _, gap, nxt, _ = _parts(site)
    nxt = bare(nxt)
    if not nxt:
        return Decision("not_ezafe", "end")
    if any(c in HARD_PUNCTUATION for c in gap) or "\n" in gap:
        return Decision("not_ezafe", "punctuation")
    for reason, words in (("connector", CONNECTORS), ("preposition", WRITTEN_PREPOSITIONS),
                          ("simile", SIMILE), ("copula", COPULAS)):
        if nxt in words:
            return Decision("not_ezafe", reason)
    if arm == "closed+verb" and any(r.verb_form != "participle" for r in readings(nxt).readings):
        return Decision("not_ezafe", "verb_reading")
    return Decision("ezafe", "default")


@lru_cache(maxsize=32768)
def word_class(token: str) -> str:
    word = bare(token)
    if not word:
        return "end"
    for name, words in (("connector", CONNECTORS | {"چون"}), ("preposition", PREPOSITIONS), ("pronoun", PRONOUNS),
                        ("copula", COPULAS), ("clitic", CLITICS)):
        if word in words:
            return name
    rs = readings(word).readings
    if any(r.verb_form != "participle" for r in rs):
        return "verb_reading"
    if rs:
        return "participle_reading"
    return "content_unknown" if has_letter(word) else "nonword"


def junction(site: dict) -> Decision:
    """Does the junction carry an ezāfe? ``ezafe``, ``none`` or ``abstain``.

    None across punctuation, before a participle's auxiliary, a connector, a
    preposition, a copula or a verb; ezāfe before a pronoun, a participle or any
    other content word. The last is a stand-in for a noun and adjective lexicon,
    not a claim that a word the verb analyser does not know is nominal.
    """
    _, head, gap, nxt, _ = _parts(site)
    if not bare(nxt) or any(unicodedata.category(c)[0] in "PS" for c in gap):
        return Decision("none", "boundary")
    if verb.constructions(readings(head), readings(nxt), separator=gap):
        return Decision("none", "participle_auxiliary")
    cls = word_class(nxt)
    if cls in ("connector", "preposition", "copula", "clitic", "verb_reading"):
        return Decision("none", cls)
    if cls in ("pronoun", "participle_reading", "content_unknown"):
        return Decision("ezafe", cls)
    return Decision("abstain", "unknown")


def score(rows, classify, target: str) -> dict:
    """Agreement with a ruling (``blind`` or ``revised``) over rows that have one."""
    confusion = {}
    for row in rows:
        if row.get(target) is not None:
            key = f"{row[target]}->{classify(row).label}"
            confusion[key] = confusion.get(key, 0) + 1
    total = sum(confusion.values())
    right = sum(n for k, n in confusion.items() if k.split("->")[0] == k.split("->")[1])
    gold = Counter(k.split("->")[0] for k, n in confusion.items() for _ in range(n))
    return {"sites": total, "agree": right, "accuracy": round(right / total, 4) if total else 0.0,
            "majority_baseline": round(max(gold.values()) / total, 4) if total else 0.0,
            "confusion": dict(sorted(confusion.items()))}


@dataclass(frozen=True, slots=True)
class Junction:
    head: str
    next: str
    span: tuple[int, int]
    mark: str
    label: str
    reason: str


def junctions(text: str) -> list[Junction]:
    """Every silent-ه junction in ``text``: ``ezafe``, ``none`` or ``abstain``, and why.
    A written mark is judged by ``written``, an unmarked junction by ``junction``."""
    out = []
    for site in sites({"text": text}):
        d = written(site) if site["marked"] else junction(site)
        out.append(Junction(site["head"], text[slice(*site["next_span"])], (site["head_span"][0], site["next_span"][1]),
                            site["observed_mark_class"], "none" if d.label == "not_ezafe" else d.label, d.reason))
    return out
