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
import math
import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from . import arabic, aruz, orthography, verb, vowels

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
# Heads that don't take an ezāfe: adverbs, and the participles of شدن and بودن.
NO_EZAFE = frozenset("همیشه چگونه چه‌گونه هرگونه هیچ‌گونه این‌گونه اینگونه آن‌گونه آنگونه شده بوده گشته گردیده".split())
PARTICIPLE_AUXILIARIES = frozenset("شده بوده گشته گردیده".split())


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
                and not any(c in text[token.end():following.start()] for c in "\n\r")
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
        replacement = neutral_head + gap
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
    if any(c in HARD_PUNCTUATION or c in "\n\r" for c in gap):
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


@lru_cache(maxsize=64)
def _scanner(metres: tuple[str, ...]) -> aruz.Scanner:
    full = aruz.scanner()
    return aruz.Scanner(aruz.Inventory(e for e in full.inventory.entries if e.label in metres), lexicon=full.lexicon,
                        devices=aruz.Devices(final_e_lengthening=False))


@lru_cache(maxsize=8)
def poem_metre(text: str) -> str | None:
    """The metre most of the text's hemistichs scan cheapest in, when there are two
    or more and at least 40% of them do. Nimaic and free verse rarely agree, and a
    line's own cheapest metre there is noise."""
    lines = aruz.hemistichs(text)[:aruz.MAX_LINES]
    found, n = max(Counter(m for line in lines for m in aruz.scanner().scan(line).metres).items(),
                   key=lambda kv: kv[1], default=(None, 0))
    return found if len(lines) >= 2 and n >= 0.4 * len(lines) else None


def metre(site: dict) -> str | None:
    """``ezafe`` or ``none``: whichever reading of the junction lets its line scan with
    fewer licences in ``site["metre"]``, or else in ``poem_metre`` of the text; None on
    a tie or with neither. A long ezāfe or -e after a silent ه is not counted as a licence:
    in the Shāhnāme's metre it is always long."""
    text = site["neutralized_text"]
    h0, h = site["neutral_head_span"]
    line = next(m for m in re.finditer(r"[^\n\t]+", text) if m.start() <= h0 < m.end())
    a, line = line.start(), line.group()
    found = site.get("metre") or poem_metre(text)
    if not found:
        return None
    scan = _scanner((found,)).scan
    cost = [s.licences - ("ezafe_lengthening" in s.devices) if s.metres else 99
            for s in (scan(line[:h - a] + "ٔ" + line[h - a:]), scan(line[:h - a] + "ْ" + line[h - a:]))]
    return None if cost[0] == cost[1] else "ezafe" if cost[0] < cost[1] else "none"


def junction(site: dict, profile: str = "prose") -> Decision:
    """Does the junction carry an ezāfe? ``ezafe``, ``none`` or ``abstain``.

    None across punctuation, before a participle's auxiliary or شده، after a head
    that takes none (همیشه، چگونه، از جمله، شده), in a reduplication (تکه تکه),
    and before a connector, a preposition, a copula or a verb. Otherwise, in
    profile ``prose``, ezāfe before a pronoun, a participle or any other content
    word. The last is a stand-in for a noun and adjective lexicon, not a claim
    that a word the verb analyser does not know is nominal. In profile ``verse``
    the metre decides (``metre``); without an agreed metre it abstains.
    """
    if profile not in ("prose", "verse"):
        raise ValueError(f"unknown profile {profile!r}")
    text, head, gap, nxt, _ = _parts(site)
    if not bare(nxt) or any(c in "\n\r" or unicodedata.category(c)[0] in "PS" for c in gap):
        return Decision("none", "boundary")
    if verb.constructions(readings(head), readings(nxt), separator=gap) or bare(nxt) in PARTICIPLE_AUXILIARIES:
        return Decision("none", "participle_auxiliary")
    if bare(head).replace(ZWNJ, "") in {w.replace(ZWNJ, "") for w in NO_EZAFE} or (
            bare(head) == "جمله" and text[:site["neutral_head_span"][0]].split()[-1:] == ["از"]):
        return Decision("none", "adverb_head")
    if bare(nxt) == bare(head):
        return Decision("none", "reduplication")
    cls = word_class(nxt)
    if cls in ("connector", "preposition", "copula", "clitic", "verb_reading"):
        return Decision("none", cls)
    after = re.match(r"(\s+)([^\s،؛؟!.:]+)", text[site["neutral_next_span"][1]:])
    if cls == "participle_reading" and after and (
            bare(after[2]) in PARTICIPLE_AUXILIARIES or verb.constructions(readings(nxt), readings(after[2]), separator=after[1])):
        return Decision("none", "participle_auxiliary")
    if profile == "verse":
        if not (site.get("metre") or poem_metre(text)):
            return Decision("abstain", "no_metre")
        return Decision(m, "metre") if (m := metre(site)) else Decision("none", "default")
    if cls in ("pronoun", "participle_reading", "content_unknown"):
        return Decision("ezafe", cls)
    return Decision("abstain", "unknown")


def score(rows, classify, target: str) -> dict:
    """Agreement with a ruling (``blind`` or ``revised``) over rows that have one."""
    confusion = {}
    for row in rows:
        if row.get(target) is not None:
            gold, predicted = ("none" if label == "not_ezafe" else label
                               for label in (row[target], classify(row).label))
            key = f"{gold}->{predicted}"
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


def junctions(text: str, profile: str = "prose", metre: str | None = None) -> list[Junction]:
    """Every silent-ه junction in ``text``: ``ezafe``, ``none`` or ``abstain``, and why.
    A written mark is judged by ``written``, an unmarked junction by ``junction``;
    ``metre`` is the poem's, for profile ``verse``."""
    out = []
    for site in sites({"text": text, "metre": metre}):
        d = written(site) if site["marked"] else junction(site, profile)
        out.append(Junction(site["head"], text[slice(*site["next_span"])], (site["head_span"][0], site["next_span"][1]),
                            site["observed_mark_class"], "none" if d.label == "not_ezafe" else d.label, d.reason))
    return out


# The ezāfe written as ه, mostly in informal writing: دیواره اتاقه من for دیوارِ اتاقِ من. The ه after a
# word is more often part of it (خونه، دسته), the copula (خوبه) or the colloquial definite (پسره رفت).
HEH_TOKEN = re.compile(rf"[\w{MARKS}{ZWNJ}]+|[^\w\s{MARKS}{ZWNJ}]")
NUMBERS = frozenset("یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده بیست سی چهل پنجاه صد هزار چند چن".split())
DEGREES = frozenset("خیلی خیلیم چقدر چقد چه انقدر اینقدر انقد اینقد کاملا واقعا یکم کمی بسیار زیاد".split())
DETERMINERS = frozenset("این اون همین همون هر یه یک هیچ کدوم کدام همه تمام".split())
SPOKEN_PREPOSITIONS = frozenset("به از با در بر برای برا واسه تو توی سر رو روی زیر بعد قبل بدون مثل مثه پیش کنار "
                                "جلو جلوی پشت تا بی".split())
NEXT = ("bound", "verb", "ra", "content", "func")
arabic_detector = lru_cache(maxsize=1)(arabic.detector)


@lru_cache(maxsize=1)
def heh_tables():
    def rows(name):
        for line in (orthography.DATA / name).read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                yield line.split("\t")
    return ({r[0]: tuple(map(int, r[1:])) for r in rows("ezafe-heh-words.tsv")},
            {(r[0], r[1]): tuple(map(int, r[2:])) for r in rows("ezafe-heh-pairs.tsv")},
            {(r[0], r[1]): tuple(map(int, r[2:])) for r in rows("ezafe-heh-left.tsv")})


@lru_cache(maxsize=1)
def heh_model():
    model = json.loads((orthography.DATA / "ezafe-heh-trees.json").read_text(encoding="utf-8"))
    for nodes, _ in model["trees"]:
        for node in nodes:
            if isinstance(node[1], list):
                node[1] = frozenset(node[1])
    return model


def heh_class(token: str | None, gap: str = " ") -> str:
    if token is None:
        return "end"
    if not gap.isspace():
        return "punct"
    if token in ("را", "رو"):
        return "ra"
    if token.isdigit() or token in NUMBERS:
        return "number"
    for name, words in (("degree", DEGREES), ("determiner", DETERMINERS), ("preposition", SPOKEN_PREPOSITIONS)):
        if token in words:
            return name
    return word_class(token)


def heh_features(head: str, nxt: str, before: str | None, after: str | None, gap: str = " ") -> dict[str, float]:
    """What the counts say about ``head`` (a word ending in ه) before ``nxt``: how edited text spells the word,
    how often the pair is written with the bare word or its kasra instead, and the classes of the words around."""
    words, pairs, left = heh_tables()
    stem, log = head[:-1], math.log
    w = words.get(head) or (0,) * 16
    spoken, edited, stem_spoken, stem_edited, kasra, suffixed = w[:6]
    f = {"unseen": float(head not in words), "lex": log((edited + 1) / (stem_edited + 1)),
         "suffixed": log((suffixed + 1) / (spoken + 1)), "kasra_stem": log((kasra + 1) / (stem_spoken + stem_edited + 1)),
         "stem_alef": float(stem[-1] in "اوآ"), "stem_yeh": float(stem[-1] in "یي")}
    for side, counts in (("head", w[6:11]), ("stem", w[11:16])):
        for name, n in zip(NEXT, counts):
            f[f"{side}_{name}"] = log((n + 0.2) / (sum(counts) + 1))
    n = [sum(c) for c in zip(*(pairs.get((stem, y), (0,) * 5) for y in {nxt, nxt[:-1] if nxt.endswith("ه") else nxt}))]
    f["pair_seen"] = float(any(n))
    f["pair_ratio"] = log((n[0] + 0.5) / (n[1] + n[2] + 0.5))
    f["pair_kasra"] = math.log1p(n[2])
    f["pair_edited"] = log((n[3] + 0.5) / (n[4] + 0.5))
    if before is not None:
        b = left.get((before, stem), (0, 0, 0))
        f["left_seen"] = float(any(b))
        f["left_ratio"] = log((b[0] + 0.5) / (b[1] + b[2] + 0.5))
    for side, word, g in (("next", nxt, " "), ("after", after, gap), ("before", before, " ")):
        f[f"{side}_{heh_class(word, g)}"] = 1.0
    return f


def heh_probability(f: dict[str, float], words: tuple[str | None, ...]) -> float:
    """Boosted trees over the features and the hashed head, next word, word before and word after, calibrated to
    the ezāfe-as-ه in 2.5% of sites."""
    model = heh_model()
    x = [f.get(n, 0.0) for n in model["features"]] + [zlib.crc32((w or "").encode()) % 4096 for w in words]
    z = 0.0
    for nodes, leaves in model["trees"]:
        i = 0
        while i >= 0:
            feature, threshold, _, _, left, right = nodes[i]
            v = x[feature]
            i = left if (v in threshold if isinstance(threshold, frozenset) else v <= threshold) else right
        z += leaves[~i]
    z = z / model["temperature"] + model["shift"]
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass(frozen=True, slots=True)
class Heh:
    first: str
    second: str
    span: tuple[int, int]
    verdict: str
    form: str
    confidence: float
    probability: float
    threshold: float
    informality: float


@lru_cache(maxsize=1)
def _register_words():
    words = heh_tables()[0]
    counts = {w: c[:2] for w, c in words.items()}
    for w, c in words.items():
        counts.setdefault(w[:-1], c[2:4])
    spoken, edited = map(sum, zip(*counts.values()))
    return {w: math.log((s + 1) / (e + 1) * edited / spoken) for w, (s, e) in counts.items() if s + e >= 10}


def informality(text: str) -> float:
    """Mean log spoken/edited frequency ratio over known words; above 1 is informal.
    Counts come from the shipped heads and stems. No known words gives 0."""
    words = _register_words()
    values = [words[w] for token in HEH_TOKEN.finditer(text) if (w := bare(token.group())) in words]
    return sum(values) / len(values) if values else 0.0


def heh(text: str, edited_floor: float = 0.9) -> list[Heh]:
    """Each word ending in ه before another word, and whether that ه is an ezāfe written as ه. ``verdict`` is
    ``wrong`` when it is, with ``form`` the pair with a kasra (ی after ا and و); ``confidence`` is the
    model's probability of that verdict, not a calibrated guarantee. Outside informal text the flagging
    threshold is ``edited_floor``; otherwise 0.5. Verbs and Arabic passages are not sites."""
    if not 0.5 <= edited_floor < 1:
        raise ValueError("edited_floor must be at least 0.5 and below 1")
    tokens = list(HEH_TOKEN.finditer(text))
    passages = arabic_detector().detect(text)
    register = informality(text)
    threshold = 0.5 if register > 1 else edited_floor
    out = []
    for i, token in enumerate(tokens[:-1]):
        head, nxt = token.group(), tokens[i + 1]
        plain_head = bare(head)
        if any(p.start <= token.start() < p.end for p in passages) or (len(plain_head.replace(ZWNJ, "")) < 3 or plain_head[-1] != "ه" or plain_head[-2] == ZWNJ or not text[token.end():nxt.start()].isspace()
                or not has_letter(nxt.group()) or word_class(head) in ("verb_reading", "participle_reading", "copula")):
            continue
        before = tokens[i - 1].group() if i and text[tokens[i - 1].end():token.start()].isspace() else None
        after, gap = (tokens[i + 2].group(), text[nxt.end():tokens[i + 2].start()]) if i + 2 < len(tokens) else (None, " ")
        plain = tuple(re.sub(f"[{MARKS}]", "", w) if w else w for w in (head, nxt.group(), before, after))
        p = heh_probability(heh_features(*plain, gap), plain)
        stem = re.sub(f"ه[{MARKS}]*$", "", head).rstrip("ِ")
        wrong = p >= threshold
        form = (stem + ("ی" if plain_head[-2] in "اوآ" else "ِ") if wrong else head) + " " + nxt.group()
        out.append(Heh(head, nxt.group(), (token.start(), nxt.end()), "wrong" if wrong else "right", form,
                       round(p if wrong else 1 - p, 3), p, threshold, round(register, 3)))
    return out
