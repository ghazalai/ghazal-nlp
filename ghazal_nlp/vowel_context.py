"""Linear context scores over word readings; inference uses only the stdlib."""

from __future__ import annotations

import gzip
import json
import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from . import vowels as v


def features(words: list[str], i: int) -> set[str]:
    out = {f"{d}:{words[i + d]}" for d in (-2, -1, 1, 2) if 0 <= i + d < len(words)}
    out.update("b:" + w for j, w in enumerate(words[max(0, i - 5):i + 6], max(0, i - 5)) if j != i)
    for d in (-1, 1):
        w = words[i + d] if 0 <= i + d < len(words) else "#"
        out.add(f"s{d}:" + w[-2:])
    for a,b in [(-2,-1),(1,2),(-1,1)]:
        if 0 <= i+a < len(words) and 0 <= i+b < len(words):
            out.add(f"p{a},{b}:"+words[i+a]+"/"+words[i+b])
    for d in (-2,2):
        if 0 <= i+d < len(words) and words[i+d//2] == "و":
            out.add("coord:"+words[i+d])
    word=words[i]
    if word in ("کرد","ترک"):
        prepositions={"از","با","به","میان","بین"}
        if i and words[i-1] in prepositions:out.add("pos:after-preposition")
        ethnic={"کرد","ترک","عرب","لر","بلوچ","فارس","مغول","ارمنی"}
        for d in (-2,2):
            j=i+d;start=min(i,j)
            if 0<=j<len(words) and words[i+d//2]=="و" and words[j] in ethnic and start>0 and words[start-1] in prepositions:
                out.add("pos:ethnic-coordination")
    if word=="کرم" and i+1<len(words):
        if words[i+1] in {"ابریشم","پیله","خاکی","نواری","حلقوی","قلابدار","ساقه‌خوار"}:out.add("compound:biology")
        if words[i+1] in {"مرطوب‌کننده","ضدآفتاب","بهداشتی","آرایشی"}:out.add("compound:cosmetic")
    if word=="ملک":
        transfer={"بخشید","بخشیدند","بخشیده","ببخشد","می‌بخشد","داد","دادند","بدهد","می‌دهد","سپرد","سپردند","بسپارد","واگذار"}
        if any(w in transfer for w in words[i+1:i+9]):
            if "را" in words[i+1:i+4] and "به" in words[i+2:i+7]:
                out.add("role:transfer-object")
                if i==0:out.add("role:initial-transfer-object")
            if i and words[i-1]=="به" and "را" in words[max(0,i-7):i]:out.add("role:transfer-recipient")
    return out


@lru_cache(maxsize=131072)
def signature(form):
    return v.editorial(form).replace(v.SUKUN, "")


def family(data, word, form):
    key = signature(form)
    for i, group in enumerate(data.get("homographs", {}).get(word, {}).get("groups", [])):
        if key in group:
            return i
    return key


def rank(data, word, record, scores, allowed):
    scale = data.get("temperature", 1.0)
    maximum = max(scores)
    ps = [math.exp(max(-700, (s - maximum) / scale)) for s in scores]
    groups = [family(data, word, f) for f in record["forms"]]
    mass = {}
    for j in allowed:
        mass[groups[j]] = mass.get(groups[j], 0) + ps[j]
    winner = max(mass, key=mass.get)
    best = max((j for j in allowed if groups[j] == winner), key=scores.__getitem__)
    return best, sum(p for p,g in zip(ps,groups) if g == winner) / sum(ps)


def alternatives(data, word, record, ambiguity, selected):
    review = data.get("homographs", {}).get(word)
    if "homographs" in data and review is None:
        return ()
    if word in ("به", "در", "بر") and signature(selected) in ("بِه", "دَر", "بَر"):
        return ()
    counts = record.get("counts", [1] * len(record["forms"]))
    grouped = {}
    for f,n in zip(record["forms"], counts):
        group = family(data, word, f)
        if review is None or isinstance(group, int):
            grouped[group] = grouped.get(group, 0) + n
    support = sorted(grouped.values(), reverse=True)
    share = support[1] / sum(support) if len(support) > 1 else 0
    share = max(share, review.get("editorial_minority", 0) if review else 0)
    chosen = family(data, word, selected)
    if len(support) < 2 or share < ambiguity or review and not isinstance(chosen, int):
        return ()
    return tuple(f for f in record["forms"] if family(data, word, f) != chosen
                 and (review is None or isinstance(family(data, word, f), int)))


def compatible(form: str, marked: str, written=False) -> bool:
    if v.skeleton(form) != v.skeleton(marked):
        return False
    if not v.is_marked(marked):return True
    left=[set(ms)&v.VOWELS for _,ms in v.slots(form)]
    right=[set(ms)&v.VOWELS for _,ms in v.slots(marked)]
    if written:
        left=[set(ms)&v.VOWELS or raw for (_,ms),raw in zip(v.slots(v.editorial(form)),left)]
        right=[set(ms)&v.VOWELS or raw for (_,ms),raw in zip(v.slots(v.editorial(marked)),right)]
    return all(not a or not b or a==b for a,b in zip(left,right))


@lru_cache(maxsize=1)
def model() -> dict:
    with gzip.open(v.DATA / "vowels-context.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=16)
def _scanner(metre):
    from . import aruz
    entries = [e for e in aruz.scanner(False).inventory.entries if e.label == metre]
    if not entries:
        raise ValueError(f"unknown metre {metre!r}")
    return aruz.Scanner(aruz.Inventory(entries))


@dataclass(frozen=True, slots=True)
class Choice:
    word: str
    span: tuple[int, int]
    form: str | None
    confidence: float
    readings: tuple[str, ...]
    reason: str


def choose(text: str, *, lexicon=None, context=None, profile="prose", metre=None) -> list[Choice]:
    """Rank each token's readings. Scores are calibrated on held-out training data.

    The default inventory is trained readings plus the word lexicon.
    A supplied lexicon augments it; untrained candidates receive no confidence.
    Written vowels constrain candidates. Verse can veto readings that do not
    scan in an explicitly supplied metre; ties retain the context scores.
    """
    if profile not in ("prose", "verse"):
        raise ValueError(f"unknown profile {profile!r}")
    data = model() if context is None else context
    lex = v.lexicon() if lexicon is None else lexicon
    matches = list(v.TOKEN.finditer(text))
    words = [v.skeleton(v.split_terminal(v.canonicalize(m.group()))[0]) for m in matches]
    out = []
    for i, (m, word) in enumerate(zip(matches, words)):
        if not v.is_persian(word):
            continue
        record = data["words"].get(word)
        entry = lex.lookup(word)
        backed = lexical_record(data, word, entry)
        trained = tuple(record["forms"]) if record else ()
        extra = list(backed.get("forms", ()))
        if entry:
            extra.extend(r.form for r in entry.readings)
        extra = tuple(dict.fromkeys(form for form in extra if form not in trained))
        forms = trained + extra
        observed = v.split_terminal(v.canonicalize(m.group()))[0]
        allowed = [j for j, form in enumerate(trained) if compatible(form, observed, written=True)]
        if not allowed:
            fallback = next((form for form in extra if compatible(form, observed, written=True)), None)
            confidence = backed.get("reliability", 0.0) if fallback and backed.get("forms", [None])[0] == fallback else 0.0
            out.append(Choice(m.group(), m.span(), fallback, confidence, forms, "lexicon" if fallback else "unknown"))
            continue
        scores = list(record["bias"])
        for f in features(words, i):
            for j, w in enumerate(record["weights"].get(f, ())):
                scores[j] += w
        reason = "context"
        if profile == "verse" and metre and len(allowed) > 1:
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            end = len(text) if end < 0 else end
            costs = {}
            for j in allowed:
                scan = _scanner(metre).scan(text[start:m.start()] + forms[j] + text[m.end():end])
                costs[j] = scan.licences if scan.metres else math.inf
            least = min(costs.values())
            finite = [j for j in allowed if math.isfinite(costs[j]) and costs[j] == least]
            if finite and len(finite) < len(allowed):
                allowed, reason = finite, "metre"
        best, p = rank(data, word, record, scores, allowed)
        # Metre and supplied marks filter choices, but cannot manufacture confidence.
        confidence = min(p, record.get("reliability", 0.0)) if len(trained) == 1 else p
        calibrated = record.get("calibration", data.get("calibration"))
        if calibrated is not None:
            confidence = min(confidence, calibrated[min(9, int(p * 10))])
        fallback = backed.get("forms", [None])[0]
        if len(trained) == 1 and confidence < .9 and backed.get("reliability", 0.0) > confidence and fallback and compatible(fallback, observed, written=True):
            out.append(Choice(m.group(), m.span(), fallback, backed["reliability"], forms, "lexicon"))
            continue
        out.append(Choice(m.group(), m.span(), forms[best], confidence, forms, reason))
    return out


def lexical_record(data, word, entry):
    if "fallback" in data:
        return data["fallback"].get(word, {})
    if not entry:
        return {}
    calibrated = data.get("lexical_reliability", {}).get(word, {})
    return {"forms": [r.form for r in entry.readings],
            "counts": [r.count for r in entry.readings],
            "reliability": calibrated.get(entry.readings[0].form, 0.0)}


def spelling(word: str, form: str, data=None, lexicon=None) -> str:
    """Full mode's spelling of a reading: editorial, sukun only where another reading has a vowel."""
    data = model() if data is None else data
    key = v.skeleton(v.canonicalize(word))
    rec = data["words"].get(key) or lexical_record(data, key, (v.lexicon() if lexicon is None else lexicon).lookup(key))
    other = [v.slots(f) for f in alternatives(data, key, rec or {}, 0, form)]
    return v._render([[ch, ms if any(i < len(f) and set(f[i][1]) & (v.VOWELS - {v.SUKUN}) for f in other)
                       else ms.replace(v.SUKUN, "")] for i, (ch, ms) in enumerate(v.slots(v.editorial(form)))])


def minimal(form: str, alternatives: tuple[str, ...], written=()) -> str:
    """The fewest attested distinguishing marks, greedily covering alternatives:
    short vowels first, the earliest on ties; sukun or shadda only for what they
    leave open, unless ``written`` lists the vowel-only spelling for this reading."""
    parsed = v.slots(form)
    others = [v.slots(f) for f in alternatives if f != form]
    uncovered = set(range(len(others)))
    marks = []
    for i, (_, ms) in enumerate(parsed):
        for mark in ms:
            if mark in v.HARAKAT:
                covers = {j for j, other in enumerate(others)
                          if i < len(other) and (set(other[i][1]) & v.VOWELS or mark == v.SHADDA
                              or i + 1 < len(other) and other[i + 1][0] in "اوی"
                              and not set(other[i + 1][1]) & v.HARAKAT
                              and mark != {"ا":v.FATHA,"و":v.DAMMA,"ی":v.KASRA}[other[i + 1][0]])
                          and mark not in other[i][1]}
                marks.append((covers, i, mark))
    chosen = [[ch, ""] for ch, _ in parsed]
    for allowed in (v.HARAKAT - {v.SUKUN, v.SHADDA}, v.HARAKAT):
        if uncovered and allowed is v.HARAKAT and v._render(chosen) in written:
            return v._render(chosen)
        while uncovered:
            covers, i, mark = max((m for m in marks if m[2] in allowed), key=lambda x: len(x[0] & uncovered),
                                  default=(set(), 0, ""))
            if not covers & uncovered:
                break
            chosen[i][1] += mark
            uncovered -= covers
    return v.skeleton(form) if uncovered else v._render(chosen)


def _overlay(original: str, form: str) -> str:
    raw = []
    for i, ch in enumerate(original):
        if ch == v.TATWEEL:
            continue
        if not unicodedata.combining(ch):
            raw.append([ch, i + 1, ""])
        elif raw:
            raw[-1][1] = i + 1
            raw[-1][2] += ch
    marked = v.slots(form)
    if len(raw) != len(marked):
        return original
    edits = []
    for a, b in zip(raw, marked):
        if v.skeleton(v.canonicalize(a[0])) != b[0]:
            return original
        added = "".join(c for c in b[1] if c in v.HARAKAT and c not in a[2]
                        and (c not in v.VOWELS or not (set(a[2]) & v.VOWELS or b[0] == "آ")))
        if added:
            edits.append((a[1], added))
    for offset, added in reversed(edits):
        original = original[:offset] + added + original[offset:]
    return original


def mark(text: str, *, mode="minimal", threshold=None, ambiguity=None, lexicon=None, context=None,
         profile="prose", metre=None, ezafe=False) -> str:
    """Add confident lexical vowels, preserving spelling, spaces and existing marks.

    ``full`` writes short vowels, shadda, tanvin and distinguishing sukun.
    ``minimal`` writes distinguishing marks on reviewed lexical heteronyms
    above the minority share (override with ``ambiguity``). Both omit redundant
    long-vowel marks and abstain below their confidence threshold.
    ``ezafe=True`` also delegates silent-heh junctions to ezafe.junctions;
    that API does not yet cover ezāfe after other letters.
    """
    if mode not in ("full", "minimal"):
        raise ValueError(f"unknown marking mode {mode!r}")
    data = model() if context is None else context
    lexicon = v.lexicon() if lexicon is None else lexicon
    threshold = data.get("minimal_threshold", .9) if threshold is None and mode == "minimal" else .9 if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    ambiguity = data.get("minimal_ambiguity", .05) if ambiguity is None else ambiguity
    if not 0 <= ambiguity <= 1:
        raise ValueError("ambiguity must be between 0 and 1")
    edits = {}
    for c in choose(text, lexicon=lexicon, context=data, profile=profile, metre=metre):
        if c.form is not None and c.confidence > 0 and c.confidence >= threshold:
            key = v.skeleton(v.canonicalize(c.word))
            rec = data["words"].get(key) or lexical_record(data, key, lexicon.lookup(key))
            if rec and rec.get("mark_reliability", 1.0) < threshold:
                continue
            if mode == "minimal":
                other = alternatives(data, key, rec or {}, ambiguity, c.form)
                if not other:
                    continue
                groups, own = data.get("homographs", {}).get(key, {}).get("groups", []), family(data, key, c.form)
                written = set(groups[own]).difference(*(g for j, g in enumerate(groups) if j != own)) if isinstance(own, int) else ()
                form = minimal(v.editorial(c.form), tuple(v.editorial(f) for f in other), written)
            else:
                form = spelling(c.word, c.form, data, lexicon)
            edits[c.span] = _overlay(c.word, form)
    if ezafe:
        from . import ezafe as e
        matches = {m.start(): m for m in v.TOKEN.finditer(text)}
        for junction in e.junctions(text, profile=profile, metre=metre):
            m = matches.get(junction.span[0])
            if m and junction.label == "ezafe" and junction.mark == "unmarked":
                edits[m.span()] = edits.get(m.span(), m.group()) + v.HAMZA
    for (a, b), replacement in sorted(edits.items(), reverse=True):
        text = text[:a] + replacement + text[b:]
    return text
