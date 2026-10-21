"""Persian verbs: analysis and generation from a stem inventory.

Every valid reading of a token is returned, each naming the rules that built
it. An unknown stem abstains. Light-verb constructions are syntax and are not
analysed; two-token tenses (رفته بودم، خواهم رفت) are read by
``constructions`` over two analysed tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

ZWNJ = "‌"
DATA = Path(__file__).parent / "data"
MI, NEG_MI, SUBJ, NEG = "می", "نمی", "ب", "ن"

PRESENT_ENDINGS = {"م": (1, "sing"), "ی": (2, "sing"), "د": (3, "sing"), "یم": (1, "plur"), "ید": (2, "plur"), "ند": (3, "plur")}
PAST_ENDINGS = {"م": (1, "sing"), "ی": (2, "sing"), "": (3, "sing"), "یم": (1, "plur"), "ید": (2, "plur"), "ند": (3, "plur")}
COPULA = {"ام": (1, "sing"), "ای": (2, "sing"), "است": (3, "sing"), "ایم": (1, "plur"), "اید": (2, "plur"), "اند": (3, "plur")}

# A bare present stem is an imperative or a noun, never a finite verb, except
# the impersonal می‌توان, which has no personal subject.
IMPERSONAL = frozenset({"توان"})
UD_ASPECT = {"imperfective": "Imp", "perfective": "Perf", "perfect": "Perf", "imperfective_perfect": "Imp"}


@dataclass(frozen=True, slots=True)
class Verb:
    infinitive: str
    past: tuple[str, ...]
    present: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reading:
    token: str
    lemma: str
    stem: str
    stem_type: str
    tense: str
    aspect: str | None
    mood: str
    person: int | None
    number: str | None
    polarity: str
    verb_form: str | None = None
    preverb: str | None = None
    base_lemma: str | None = None
    rules: tuple[str, ...] = ()

    def features(self) -> str:
        """UD-shaped features, comparable with treebank FEATS."""
        parts = [f"Tense={'Past' if self.tense == 'past' else 'Pres'}"]
        if self.aspect:
            parts.append(f"Aspect={UD_ASPECT[self.aspect]}")
        parts.append(f"Mood={self.mood.title()}")
        if self.person:
            parts.append(f"Person={self.person}")
        if self.number:
            parts.append(f"Number={'Sing' if self.number == 'sing' else 'Plur'}")
        parts.append(f"Polarity={'Neg' if self.polarity == 'neg' else 'Pos'}")
        if self.verb_form:
            parts.append(f"VerbForm={'Part' if self.verb_form == 'participle' else 'Fin'}")
        if self.aspect == "imperfective_perfect":
            parts.append("Construction=ImperfectivePerfect")
        return "|".join(parts)


@dataclass(frozen=True, slots=True)
class Analysis:
    token: str
    readings: tuple[Reading, ...]
    abstention: str | None = None


class Lexicon:
    def __init__(self, verbs=()):
        self.verbs = list(verbs)
        self.by_past: dict[str, list[Verb]] = {}
        self.by_present: dict[str, list[Verb]] = {}
        self.by_infinitive = {}
        for verb in self.verbs:
            for stem in verb.past:
                self.by_past.setdefault(stem, []).append(verb)
            for stem in verb.present:
                self.by_present.setdefault(stem, []).append(verb)
            self.by_infinitive.setdefault(verb.infinitive, verb)

    def __len__(self) -> int:
        return len(self.verbs)

    @classmethod
    def load(cls, path: Path = DATA / "verb-stems.tsv") -> Lexicon:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
        return cls(Verb(i.strip(), tuple(p.strip().split("|")), tuple(s.strip().split("|"))) for i, p, s in rows)


def load_preverbs(path: Path = DATA / "verb-preverbs.tsv") -> list[tuple[str, str]]:
    """Attested (preverb, base infinitive) pairs."""
    return [tuple(line.split("\t")[:2]) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def _split_ending(body: str, endings):
    for ending in sorted(endings, key=len, reverse=True):
        if ending == "":
            yield body, ""
        elif body.endswith(ending) and len(body) > len(ending):
            yield body[:-len(ending)], ending


def _stem_allowed(stem: str, alternates, *, prefixed: bool) -> bool:
    """A listed alternate that drops an initial ا (افت → فت) or turns آ into یا
    (آور → یاور) stands only after ب, ن or می."""
    return prefixed or not any(
        (base.startswith("ا") and stem == base[1:]) or (base.startswith("آ") and stem == "یا" + base[1:])
        for base in alternates
    )


class Analyzer:
    def __init__(self, lexicon: Lexicon | None = None, preverbs=None):
        self.lexicon = lexicon if lexicon is not None else Lexicon.load()
        self.preverbs: dict[str, set[str]] = {}
        for preverb, base in load_preverbs() if preverbs is None else preverbs:
            if base in self.lexicon.by_infinitive:
                self.preverbs.setdefault(preverb, set()).add(base)

    def analyze(self, token: str) -> Analysis:
        token = token.strip()
        if not token:
            return Analysis(token, ())
        readings = [r for polarity, aspect, mood, body in self._prefixes(token)
                    for r in self._body(token, body, polarity, aspect, mood, prefixed=body != token)]
        readings += self._compounds(token)
        if readings:
            readings = list(dict.fromkeys(readings))
            readings.sort(key=lambda r: (r.lemma, r.tense, r.mood, r.person or 0))
            return Analysis(token, tuple(readings))
        if any(t in token for t in ("یا", "ین", "یف")) and token.startswith((SUBJ, NEG, MI)):
            return Analysis(token, (), "epenthesis_not_modelled")
        return Analysis(token, (), "unknown_stem")

    def _compounds(self, token: str):
        """An attested preverb before the inflected base (برنمی‌دارد) or after
        می (می‌بردارد), with at most one ZWNJ at its edge."""
        for preverb, bases in self.preverbs.items():
            paths = []
            if token.startswith(preverb):
                rest = token[len(preverb):].removeprefix(ZWNJ)
                paths += [(*p, p[3] != rest, "preverb_before_inflection") for p in self._prefixes(rest)]
            for polarity, aspect, mood, body in self._prefixes(token):
                if aspect == "imperfective" and body.startswith(preverb):
                    paths.append((polarity, aspect, mood, body[len(preverb):].removeprefix(ZWNJ), True, "mi_before_preverb"))
            for polarity, aspect, mood, body, prefixed, rule in paths:
                bare = aspect is None and mood == "indicative" and polarity == "pos"
                parsed = list(self._body(token, body, polarity, aspect, mood, prefixed=prefixed, bare_imperative=bare))
                if bare:  # a compound subjunctive may omit ب
                    parsed += [replace(r, mood="subjunctive") for r in tuple(parsed)
                               if r.stem_type == "present" and r.mood == "indicative"]
                for r in parsed:
                    if r.lemma in bases:
                        yield replace(r, lemma=preverb + r.lemma, preverb=preverb, base_lemma=r.lemma,
                                      rules=(*r.rules, f"separable_preverb:{rule}"))

    def _prefixes(self, token: str):
        """(polarity, aspect, mood, body) for every prefix reading."""
        if token.startswith(NEG_MI + ZWNJ):
            yield "neg", "imperfective", "indicative", token[len(NEG_MI) + 1:]
            return
        if token.startswith(MI + ZWNJ):
            yield "pos", "imperfective", "indicative", token[len(MI) + 1:]
            return
        yield "pos", None, "indicative", token
        if token.startswith(SUBJ) and len(token) > 2:
            yield "pos", None, "subjunctive", token[1:]
        if token.startswith(NEG) and len(token) > 2:
            yield "neg", None, "indicative", token[1:]
            yield "neg", None, "subjunctive", token[1:]

    def _body(self, token, body, polarity, aspect, mood, *, prefixed=False, bare_imperative=False):
        past, present = self.lexicon.by_past, self.lexicon.by_present
        if ZWNJ in body:  # the perfect: رفته‌ام
            head, _, tail = body.rpartition(ZWNJ)
            if tail in COPULA and head.endswith("ه"):
                for verb in past.get(head[:-1], ()):
                    yield Reading(token, verb.infinitive, head[:-1], "past", "past",
                                  "imperfective_perfect" if aspect == "imperfective" else "perfect",
                                  mood, *COPULA[tail], polarity, "finite" if aspect == "imperfective" else None,
                                  rules=("perfect",))
            return
        if body.endswith("ه") and (aspect == "imperfective" or (aspect is None and mood == "indicative")):
            for verb in past.get(body[:-1], ()):
                yield Reading(token, verb.infinitive, body[:-1], "past", "past",
                              "imperfective_perfect" if aspect else "perfect", mood, None, None, polarity,
                              "participle", rules=("mi_participle" if aspect else "past_participle",))
        if mood != "subjunctive":  # ب does not attach to a past stem
            for stem, ending in _split_ending(body, PAST_ENDINGS):
                for verb in past.get(stem, ()):
                    yield Reading(token, verb.infinitive, stem, "past", "past", aspect or "perfective",
                                  "indicative", *PAST_ENDINGS[ending], polarity)
        for stem, ending in _split_ending(body, PRESENT_ENDINGS):
            for verb in present.get(stem, ()):
                # A stem with a glide alternate takes it before ی: می‌گوید is گوی+د, not گو+ید.
                if not _stem_allowed(stem, verb.present, prefixed=prefixed) or (
                        ending.startswith("ی") and stem + "ی" in verb.present):
                    continue
                yield Reading(token, verb.infinitive, stem, "present", "present", aspect, mood,
                              *PRESENT_ENDINGS[ending], polarity)
                if ending == "ید" and aspect is None and (mood == "subjunctive" or bare_imperative):
                    yield Reading(token, verb.infinitive, stem, "present", "present", None, "imperative",
                                  2, "plur", polarity, rules=("imperative",))
        if body in IMPERSONAL and aspect is not None:
            for verb in present.get(body, ()):
                yield Reading(token, verb.infinitive, body, "present", "present", aspect, mood, 3, "sing", polarity)
        if mood == "subjunctive" or bare_imperative:
            for verb in present.get(body, ()):
                if _stem_allowed(body, verb.present, prefixed=prefixed):
                    yield Reading(token, verb.infinitive, body, "present", "present", None, "imperative",
                                  2, "sing", polarity)

    def conjugate(self, infinitive: str, *, tense="present", mood="indicative", person=3, number="sing",
                  polarity="pos", aspect=None) -> str | None:
        """A finite form, or None when the paradigm has none or it needs epenthesis."""
        verb = self.lexicon.by_infinitive.get(infinitive)
        if verb is None:
            for preverb, bases in self.preverbs.items():
                base = infinitive.removeprefix(preverb)
                if base == infinitive or base not in bases:
                    continue
                form = self.conjugate(base, tense=tense, mood=mood, person=person, number=number,
                                      polarity=polarity, aspect=aspect)
                if form is None:
                    return None
                if tense == "present" and mood in ("subjunctive", "imperative") and polarity == "pos":
                    form = form.removeprefix(SUBJ)
                return preverb + form
            return None
        cell = (person, number)
        if cell not in COPULA.values():
            return None
        mi = NEG_MI if polarity == "neg" else MI
        if tense == "past":
            if aspect == "imperfective_perfect":
                copula = next(e for e, v in COPULA.items() if v == cell)
                return mi + ZWNJ + verb.past[0] + "ه" + ZWNJ + copula
            body = verb.past[0] + next(e for e, v in PAST_ENDINGS.items() if v == cell)
            if aspect == "imperfective":
                return mi + ZWNJ + body
            return NEG + body if polarity == "neg" else body
        ending = next(e for e, v in PRESENT_ENDINGS.items() if v == cell)
        stem = verb.present[0]
        if stem.startswith(("آ", "ا", "ی")) and mood in ("subjunctive", "imperative"):
            return None  # بیایم، بینداز: the glide and the deleted alef
        prefix = NEG if polarity == "neg" else SUBJ
        if mood == "imperative":
            return prefix + stem + ("" if number == "sing" else "ید")
        if stem + "ی" in verb.present:
            stem += "ی"
        return prefix + stem + ending if mood == "subjunctive" else mi + ZWNJ + stem + ending


@lru_cache(maxsize=1)
def analyzer() -> Analyzer:
    return Analyzer()


@dataclass(frozen=True, slots=True)
class Construction:
    """A possible two-token tense; adjectival participles and lexical شدن or
    خواستن can look the same, and negation scope is not resolved."""

    tokens: tuple[str, str]
    lemma: str
    construction: str
    tense: str
    aspect: str | None
    mood: str
    person: int
    number: str
    polarity: str
    voice: str
    component_polarities: tuple[str, str]
    components: tuple[Reading, ...]


def _auxiliaries():
    forms = {form: ("perfect", "indicative", p, n, "pos") for form, (p, n) in COPULA.items()}
    for ending, (p, n) in PAST_ENDINGS.items():
        forms["نیست" + ending] = ("perfect", "indicative", p, n, "neg")
        for prefix, polarity in (("", "pos"), ("ن", "neg")):
            forms[prefix + "بود" + ending] = ("past_perfect", "indicative", p, n, polarity)
    for prefix, polarity in (("", "pos"), ("ن", "neg")):
        forms[prefix + "باش"] = ("perfect_imperative", "imperative", 2, "sing", polarity)
        for ending, (p, n) in PRESENT_ENDINGS.items():
            forms[prefix + "باش" + ending] = ("perfect_subjunctive", "subjunctive", p, n, polarity)
    return forms


AUXILIARIES = _auxiliaries()


def constructions(left: Analysis, right: Analysis, *, separator: str = " ") -> tuple[Construction, ...]:
    """Perfects, past perfects, passives and futures over two adjacent tokens
    separated by whitespace only. The token readings are not changed."""
    if not separator or not separator.isspace():
        return ()
    out = []

    def add(head, aux, kind, tense, aspect, mood, person, number, aux_polarity, voice="active", head_first=True):
        polarities = (head.polarity, aux_polarity) if head_first else (aux_polarity, head.polarity)
        components = (head,) + ((aux,) if aux else ())
        out.append(Construction((left.token, right.token), head.lemma, kind, tense, aspect, mood, person, number,
                                "neg" if "neg" in polarities else "pos", voice, polarities,
                                components if head_first else components[::-1]))

    for head in left.readings:
        if head.verb_form != "participle":
            continue
        if aux := AUXILIARIES.get(right.token):
            kind, mood, person, number, polarity = aux
            add(head, None, kind, "past", head.aspect, mood, person, number, polarity)
            if right.token in ("باشید", "نباشید"):
                add(head, None, "perfect_imperative", "past", head.aspect, "imperative", 2, "plur", polarity)
        if head.aspect == "perfect":
            for aux in right.readings:
                if aux.lemma == "شدن" and aux.verb_form != "participle" and aux.person is not None:
                    add(head, aux, "passive", aux.tense, aux.aspect, aux.mood, aux.person, aux.number,
                        aux.polarity, "passive")
    for aux in left.readings:
        if not (aux.lemma == "خواستن" and aux.stem == "خواه" and aux.tense == "present"
                and aux.mood == "indicative" and aux.aspect is None and aux.person is not None):
            continue
        for head in right.readings:
            # The short infinitive is the bare past stem, so only that surface is a future.
            if (head.tense == "past" and head.aspect == "perfective" and head.polarity == "pos" and head.person == 3
                    and head.number == "sing" and right.token.replace(ZWNJ, "") == (head.preverb or "") + head.stem):
                add(head, aux, "future", "future", None, "indicative", aux.person, aux.number, aux.polarity,
                    head_first=False)
    return tuple(dict.fromkeys(out))
