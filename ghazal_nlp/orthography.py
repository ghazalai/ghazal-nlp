"""Persian spelling rules, checked over the text as written.

Two styles: ``academy``, the Academy's دستور خطّ فارسی with its pages, and
``ghazal``, the house style, which adopts the Academy rules careful writers
follow and has its own where they don't. A rule ships correct Persian it must
not flag and wrong Persian it must flag, and doesn't load if it fails them. A
case a rule can't decide comes back as an abstention with a reason.
"""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from . import verb

RULEBASE_VERSION = "0.2.0"
RULES = Path(__file__).parent / "rules"
DATA = Path(__file__).parent / "data"

ZWNJ = "‌"
# One Persian word: \w misses the harakat, hence the Arabic block; ZWNJ and ZWJ
# join a word only between two word characters.
WORD = re.compile(r"[\w؀-ۿ](?:[\w؀-ۿ‌‍]*[\w؀-ۿ])?")
PERSIAN = re.compile(r"^[ء-ۿ‌]+$")
SPACES = re.compile(r"^[ \t\xa0]+$")
# Persian punctuation sits in the Arabic block, so «واقعا،» is one word.
EDGE_PUNCTUATION = "،؛؟؍؞٪٫٬٭۔"
# Letters that do not join the next one: after them a compound needs no ZWNJ (p. 23).
NON_JOINING = frozenset("اآأإدذرزژوؤءةى")
# A line in another Arabic-script language: a letter Persian lacks, or a Kurdish spelling.
OTHER_SCRIPT = re.compile(r"[ڵڕۆێڤۊٹڈڑںےۓټډړږښګڼېڻڦڄڃڇڍڊڌڎڏۈۋ]|(?<![^\s«(\"'])ئ|وو|یەک|ەکان|ەکە|(?<!\S)لە(?!\S)")
ARABIC_LETTER = re.compile(r"[\u0621-\u064a\u066e-\u06d3\u06d5]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
LINES = {
    "persian_script": lambda line: not OTHER_SCRIPT.search(line),
    "persian": lambda line: len(ARABIC_LETTER.findall(line)) > len(LATIN_LETTER.findall(line)),
}

LAYERS = frozenset({"orthography", "phonology", "prosody", "morphology", "syntax", "lexicon", "discourse"})
STATUSES = frozenset({"proposed", "active", "contested", "retired"})
STYLES = ("ghazal", "academy")
ACADEMY_POSITIONS = frozenset({"right", "wrong", "either", "unspecified", "varies"})
KINDS = frozenset({"prescriptive", "descriptive", "contested"})
REGISTERS = frozenset({"formal_written", "colloquial", "verse", "classical", "technical"})
RUNNING = frozenset({"active", "contested"})
CONFIDENCE = {"low": 0, "medium": 1, "high": 2}


@lru_cache(maxsize=1)
def silent_he_inventory() -> frozenset[str]:
    """The words proven to end in a silent ه."""
    lines = (DATA / "orthography-silent-he.tsv").read_text(encoding="utf-8").splitlines()
    return frozenset(line.split("\t")[0].replace(ZWNJ, "") for line in lines if line.strip() and not line.startswith("#"))


def is_silent_he(word: str) -> bool | None:
    """Is the final ه of this bare word silent? None when the evidence cannot say.

    A final ه after ا is pronounced (ماه، راه): of 139 such stems with enough
    evidence only نشأه is silent. Two-letter words always abstain.
    """
    bare = word.replace(ZWNJ, "")
    if not bare.endswith("ه") or len(bare) < 3:
        return None
    if bare.endswith("اه"):
        return False
    return True if bare in silent_he_inventory() else None


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def bare(self) -> str:
        return self.text.replace(ZWNJ, "")


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]


@dataclass(frozen=True, slots=True)
class Finding:
    rule_id: str
    layer: str
    kind: str
    span: tuple[int, int]
    text: str
    suggestion: str | None
    severity: str = "violation"
    confidence: str = "high"
    citation: str = ""
    contested: bool = False
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Abstention:
    """A case the rule saw and declined to judge."""

    rule_id: str
    reason: str
    span: tuple[int, int]
    text: str


@dataclass(frozen=True, slots=True)
class AffixZwnj:
    """A prefix or suffix bound to its host with ZWNJ, written instead with a
    space (strong) or solid (weak: دختر is not دخ + تر, so the solid finding is
    advisory and an exception that is a stem exempts its longer forms)."""

    rule_id: str
    layer: str
    affix: str
    position: str = "suffix"
    variants: tuple[str, ...] = ()
    detect_space: bool = True
    detect_solid: bool = True
    solid_exceptions: frozenset[str] = frozenset()
    lexical_exceptions: frozenset[str] = frozenset()
    require_host_in: frozenset[str] | None = None
    host_exceptions: frozenset[str] = frozenset()
    solid_hosts: str | None = None
    free_form: bool = False
    min_host: int = 3
    space_severity: str = "violation"
    solid_severity: str = "advisory"

    @property
    def forms(self) -> tuple[str, ...]:
        return tuple(sorted({self.affix, *self.variants}, key=len, reverse=True))

    def _exempt(self, word: str) -> bool:
        exceptions = self.solid_exceptions | self.lexical_exceptions
        return word in exceptions or any(word.startswith(e) and len(word) > len(e) for e in exceptions)

    def scan(self, text: str, tokens: Sequence[Token]) -> Iterator[Finding | Abstention]:
        if self.detect_space:
            yield from self._scan_space(text, tokens)
        if self.detect_solid:
            yield from self._scan_solid(tokens)

    def _scan_space(self, text, tokens):
        suffix = self.position == "suffix"
        for i, token in enumerate(tokens):
            if token.text not in self.forms:
                continue
            j = i - 1 if suffix else i + 1
            if not 0 <= j < len(tokens):
                continue
            host = tokens[j]
            if not SPACES.match(text[host.end:token.start] if suffix else text[token.end:host.start]):
                continue
            if host.bare in self.host_exceptions:
                continue
            span = (min(host.start, token.start), max(host.end, token.end))
            if not PERSIAN.match(host.text) or len(host.bare) < self.min_host:
                yield Abstention(self.rule_id, "host_too_short_or_not_persian", span, text[span[0]:span[1]])
            elif self.require_host_in is not None and host.bare not in self.require_host_in:
                yield Abstention(self.rule_id, "host_not_in_compound_lexicon", span, text[span[0]:span[1]])
            else:
                yield Finding(
                    self.rule_id, self.layer, "space_where_zwnj", span, text[span[0]:span[1]],
                    host.text + ZWNJ + token.text if suffix else token.text + ZWNJ + host.text,
                    self.space_severity, "medium" if self.free_form else "high",
                    note=f"«{self.affix}» also occurs as a free word; the reading is not certain"
                    if self.free_form else None,
                )

    def _scan_solid(self, tokens):
        hosts = closed_list(self.solid_hosts) if self.solid_hosts else None
        for token in tokens:
            if ZWNJ in token.text:
                continue
            word = token.text
            for form in self.forms:
                if self.position == "suffix":
                    if not word.endswith(form):
                        continue
                    host = word[:-len(form)]
                elif not word.startswith(form):
                    continue
                else:
                    host = word[len(form):]
                if host and not self._exempt(word):
                    if len(host) < self.min_host or not PERSIAN.match(host):
                        yield Abstention(self.rule_id, "host_too_short_to_decide", (token.start, token.end), word)
                    elif hosts is not None and host not in hosts:
                        yield Abstention(self.rule_id, "host_not_attested", (token.start, token.end), word)
                    else:
                        yield Finding(
                            self.rule_id, self.layer, "solid_where_zwnj", (token.start, token.end), word,
                            host + ZWNJ + form if self.position == "suffix" else form + ZWNJ + host,
                            self.solid_severity, "medium",
                            note="exception lexicon is incomplete until فرهنگ املایی خطّ فارسی is acquired",
                        )
                break


@dataclass(frozen=True, slots=True)
class Pattern:
    """A regex, for what is decidable from characters alone. ``repair`` is a
    ``re.sub`` template, or ``translate`` maps characters of the match;
    ``lines`` keeps only the lines a ``LINES`` test accepts."""

    rule_id: str
    layer: str
    pattern: str
    kind: str = "pattern"
    repair: str | None = None
    translate: dict | None = None
    lines: str | None = None
    severity: str = "violation"
    confidence: str = "high"
    note: str | None = None

    def __post_init__(self):
        if self.lines is not None and self.lines not in LINES:
            raise TypeError(f"unknown lines test {self.lines!r}")

    def _suggest(self, m: re.Match) -> str | None:
        if self.translate is not None:
            return m.group().translate(str.maketrans(self.translate))
        return m.expand(self.repair) if self.repair is not None else None

    def scan(self, text: str, tokens: Sequence[Token]) -> Iterator[Finding]:
        start = 0
        for line in text.split("\n") if self.lines else [text]:
            if not self.lines or LINES[self.lines](line):
                for m in re.finditer(self.pattern, line):
                    yield Finding(self.rule_id, self.layer, self.kind, (start + m.start(), start + m.end()), m.group(),
                                  self._suggest(m), self.severity, self.confidence, note=self.note)
            start += len(line) + 1


SPELLINGS = {"spaced": " ", "zwnj": ZWNJ, "solid": ""}


def _spell(first: str, second: str, form: str) -> str:
    if form == "zwnj" and first[-1] in NON_JOINING:
        form = "solid"
    return first + SPELLINGS[form] + second


@lru_cache(maxsize=None)
def closed_list(name: str) -> dict[str, str]:
    """A ``wrong<TAB>right`` table in ``data/``, or a pair table: first word,
    second word, the house spelling, the other spellings (spaced, zwnj, solid)
    and any evidence after them."""
    table = {}
    for line in (DATA / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) >= 4 and cells[2] in SPELLINGS:
            first, second, house, others = cells[:4]
            right = _spell(first, second, house)
            for other in others.split(","):
                if (wrong := _spell(first, second, other)) != right:
                    table.setdefault(wrong, right)
        else:
            table[cells[0]] = cells[1] if len(cells) > 1 else ""
    return table


@lru_cache(maxsize=None)
def _index(file, entries, abstain_on) -> dict[str, list[tuple[tuple[str, ...], str | None]]]:
    table: dict[str, str | None] = {**(closed_list(file) if file else {}), **dict(entries)}
    table.update(dict.fromkeys(abstain_on))
    index: dict[str, list] = {}
    for key, repair in table.items():
        parts = tuple(key.split())
        index.setdefault(parts[0], []).append((parts, repair))
    for group in index.values():
        group.sort(key=lambda item: len(item[0]), reverse=True)
    return index


def _core(token: Token) -> tuple[str, int, int]:
    start = token.start + len(token.text) - len(token.text.lstrip(EDGE_PUNCTUATION))
    core = token.text.strip(EDGE_PUNCTUATION)
    return core, start, start + len(core)


@dataclass(frozen=True, slots=True)
class ClosedList:
    """Forms a standard rejects, each with the form it requires. Entries may be
    phrases over tokens separated by spaces only. Matching is exact: اطاق is
    caught, اطاق‌ها is not. ``abstain_on`` holds the entries the standard leaves
    to sense (هر چند)."""

    rule_id: str
    layer: str
    entries: dict[str, str] = field(default_factory=dict)
    file: str | None = None
    abstain_on: frozenset[str] = frozenset()
    abstain_reason: str = "the standard makes this one sense-dependent"
    abstain_before: str | None = None
    kind: str = "non_standard_form"
    severity: str = "violation"
    confidence: str = "high"
    note: str | None = None

    def _candidates(self) -> dict[str, list[tuple[tuple[str, ...], str | None]]]:
        return _index(self.file, tuple(self.entries.items()), self.abstain_on)

    def scan(self, text: str, tokens: Sequence[Token]) -> Iterator[Finding | Abstention]:
        candidates = self._candidates()
        cores = [_core(t) for t in tokens]
        i = 0
        while i < len(tokens):
            for parts, repair in candidates.get(cores[i][0], ()):
                last = i + len(parts) - 1
                if last >= len(tokens) or any(cores[i + k][0] != parts[k] for k in range(len(parts))):
                    continue
                if any(not SPACES.match(text[cores[i + k][2]:cores[i + k + 1][1]]) for k in range(len(parts) - 1)):
                    continue
                span = (cores[i][1], cores[last][2])
                found = text[span[0]:span[1]]
                if self.abstain_before and last + 1 < len(tokens) and cores[last + 1][0].startswith(self.abstain_before):
                    yield Abstention(self.rule_id, f"before {self.abstain_before}", span, found)
                elif repair is None:
                    yield Abstention(self.rule_id, self.abstain_reason, span, found)
                else:
                    yield Finding(self.rule_id, self.layer, self.kind, span, found, repair,
                                  self.severity, self.confidence, note=self.note)
                i = last
                break
            i += 1


def _silent_gate(rule_id: str, host: str, span, found) -> Abstention | None:
    silent = is_silent_he(host)
    if silent is None:
        return Abstention(rule_id, "he_not_proven_silent", span, found)
    if silent is False:
        return Abstention(rule_id, "he_is_voiced", span, found)
    return None


@dataclass(frozen=True, slots=True)
class Enclitic:
    """A clitic written with a space where the tables of pp. 42–46 join it: the
    clitic takes a ZWNJ exactly when the host ends in ه، ی or و. A host in ه
    counts only when its ه is proven silent, or «بده ای دوست» would bond a
    vocative onto an imperative."""

    rule_id: str
    layer: str
    clitics: frozenset[str]
    hosts: frozenset[str] = frozenset({"ه"})
    require_silent_he: bool = True
    min_host: int = 3
    kind: str = "space_where_zwnj"
    severity: str = "violation"
    confidence: str = "high"
    note: str | None = None

    def scan(self, text: str, tokens: Sequence[Token]) -> Iterator[Finding | Abstention]:
        for token, clitic in zip(tokens, tokens[1:]):
            if clitic.text not in self.clitics or not SPACES.match(text[token.end:clitic.start]):
                continue
            host = token.bare
            span = (token.start, clitic.end)
            found = text[span[0]:span[1]]
            if not host or host[-1] not in self.hosts:
                continue
            if len(host) < self.min_host or not PERSIAN.match(token.text):
                yield Abstention(self.rule_id, "host_too_short_or_not_persian", span, found)
                continue
            gate = host.endswith("ه") and self.require_silent_he and _silent_gate(self.rule_id, host, span, found)
            if gate:
                yield gate
                continue
            yield Finding(self.rule_id, self.layer, self.kind, span, found, token.text + ZWNJ + clitic.text,
                          self.severity, self.confidence, note=self.note)


def _join(host: str, second: str) -> str:
    bare = host.replace(ZWNJ, "")
    return host + second if bare and bare[-1] in NON_JOINING else host + ZWNJ + second


@dataclass(frozen=True, slots=True)
class Compound:
    """A compound written with a space where the compounds chapter joins it. The
    second element decides, and it is a closed set the standard prints, so an
    ezāfe phrase («خانهٔ من») is never joined. ``solid_exceptions`` are compounds
    written solid; ``host_exceptions`` first words never a compound's first part
    (همه، جمله)."""

    rule_id: str
    layer: str
    seconds: frozenset[str]
    require_silent_he: bool = False
    solid_exceptions: frozenset[str] = frozenset()
    host_exceptions: frozenset[str] = frozenset()
    min_host: int = 3
    kind: str = "space_where_zwnj"
    severity: str = "violation"
    confidence: str = "high"
    note: str | None = None

    def scan(self, text: str, tokens: Sequence[Token]) -> Iterator[Finding | Abstention]:
        for token, second in zip(tokens, tokens[1:]):
            if second.bare not in self.seconds or not SPACES.match(text[token.end:second.start]):
                continue
            host = token.bare
            if host in self.host_exceptions:
                continue
            span = (token.start, second.end)
            found = text[span[0]:span[1]]
            if len(host) < self.min_host or not PERSIAN.match(token.text):
                yield Abstention(self.rule_id, "host_too_short_or_not_persian", span, found)
                continue
            if self.require_silent_he:
                if not host.endswith("ه"):
                    continue
                if gate := _silent_gate(self.rule_id, host, span, found):
                    yield gate
                    continue
            solid = host + second.bare
            listed = solid in self.solid_exceptions
            yield Finding(
                self.rule_id, self.layer, "space_where_joined" if listed else self.kind, span, found,
                solid if listed else _join(token.text, second.text), self.severity, self.confidence,
                note="the standard lists this compound as written solid" if listed else self.note,
            )


@dataclass(frozen=True, slots=True)
class VerbPrefix:
    """می or نمی written spaced or solid before a verb. The verb analyser
    decides, so میدان and میوه are left alone; ``exceptions`` are solid forms that
    are also words (میدانی)."""

    rule_id: str
    layer: str
    prefixes: tuple[str, ...] = ("نمی", "می")
    exceptions: frozenset[str] | str = frozenset()
    kind: str = "verb_prefix_without_zwnj"
    severity: str = "violation"
    confidence: str = "high"
    note: str | None = None

    def _skip(self) -> frozenset[str]:
        if isinstance(self.exceptions, str):
            return frozenset(closed_list(self.exceptions))
        return self.exceptions

    def scan(self, text: str, tokens: Sequence[Token]) -> Iterator[Finding]:
        skip = self._skip()
        for i, token in enumerate(tokens):
            if token.text in self.prefixes and i + 1 < len(tokens) and SPACES.match(text[token.end:tokens[i + 1].start]):
                body, span = tokens[i + 1].text, (token.start, tokens[i + 1].end)
                prefix = token.text
            elif ZWNJ not in token.text and token.text not in skip:
                prefix = next((p for p in self.prefixes if token.text.startswith(p)), None)
                if prefix is None:
                    continue
                body, span = token.text[len(prefix):], (token.start, token.end)
            else:
                continue
            joined = prefix + ZWNJ + body
            if len(body) >= 2 and PERSIAN.match(body) and verb.analyzer().analyze(joined).readings:
                yield Finding(self.rule_id, self.layer, self.kind, span, text[span[0]:span[1]], joined,
                              self.severity, self.confidence, note=self.note)


DETECTORS = {"affix_zwnj": AffixZwnj, "closed_list": ClosedList, "compound": Compound,
             "enclitic": Enclitic, "pattern": Pattern, "verb_prefix": VerbPrefix}


class RuleError(ValueError):
    """A malformed rule record, or a rule that fails its own examples."""


@dataclass(frozen=True, slots=True)
class Source:
    work: str
    page: str | None = None
    evidence: str | None = None
    publisher: str | None = None
    edition: str | None = None
    isbn: str | None = None
    locator: str | None = None
    retrieved: str | None = None
    sha256: str | None = None

    def citation(self) -> str:
        parts = [self.work, self.publisher, self.edition, self.page and f"ص {self.page}", self.locator]
        return "، ".join(p for p in parts if p)


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    expect: str | None = None


@dataclass(frozen=True)
class Rule:
    id: str
    layer: str
    title: dict[str, str]
    statement: dict[str, str]
    kind: str
    status: str
    registers: tuple[str, ...]
    source: Source
    examples_ok: tuple[str, ...]
    examples_bad: tuple[Example, ...]
    detector: object | None = None
    severity: str = "violation"
    confidence: str = "high"
    notes: str | None = None
    disputes: tuple[dict[str, str], ...] = ()
    academy: dict[str, str] | None = None

    @property
    def style(self) -> str:
        return self.id.split(".")[0]

    @property
    def executable(self) -> bool:
        return self.detector is not None

    @property
    def runs(self) -> bool:
        return self.executable and self.status in RUNNING

    @property
    def contested(self) -> bool:
        """The sources disagree: the rule runs, and its findings say so."""
        return self.status == "contested" or self.kind == "contested"

    def applies_in(self, registers: Iterable[str]) -> bool:
        return bool(set(self.registers) & set(registers))

    def scan(self, text: str) -> tuple[list[Finding], list[Abstention]]:
        findings, abstentions = [], []
        if self.detector is not None:
            citation = self.source.citation()
            for r in self.detector.scan(text, tokenize(text)):
                if isinstance(r, Abstention):
                    abstentions.append(r)
                else:
                    findings.append(replace(r, citation=citation, contested=self.contested))
        return findings, abstentions

    def verify(self) -> None:
        """Raise unless the rule passes its own examples: a flagged ``ok`` is too
        broad, a missed violation or repair too narrow."""
        if not self.executable:
            return
        for text in self.examples_ok:
            findings, _ = self.scan(text)
            if findings:
                raise RuleError(f"{self.id}: correct example «{text}» was flagged "
                                f"({findings[0].kind} at {findings[0].span})")
        for example in self.examples_bad:
            findings, abstentions = self.scan(example.text)
            if not findings:
                reason = abstentions[0].reason if abstentions else "no finding"
                raise RuleError(f"{self.id}: violation «{example.text}» not caught ({reason})")
            got = {f.suggestion for f in findings}
            if example.expect is not None and example.expect not in got:
                raise RuleError(f"{self.id}: violation «{example.text}» expected repair "
                                f"«{example.expect}», got {sorted(s for s in got if s)}")


def _require(record: dict, key: str, rule_id: str):
    if key not in record or record[key] in (None, "", [], {}):
        raise RuleError(f"{rule_id}: missing required field '{key}'")
    return record[key]


def _detector(spec: dict, rule_id: str, layer: str):
    if spec.get("type") not in DETECTORS:
        raise RuleError(f"{rule_id}: unknown detector type {spec.get('type')!r}")
    params = {k: (tuple(v) if k == "variants" else frozenset(v)) if isinstance(v, list) else v
              for k, v in spec.items() if k != "type"}
    try:
        return DETECTORS[spec["type"]](rule_id=rule_id, layer=layer, **params)
    except TypeError as e:
        raise RuleError(f"{rule_id}: bad detector parameters — {e}") from e


def parse_rule(record: dict) -> Rule:
    rule_id = record.get("id") or "<unnamed>"
    for key, allowed in (("layer", LAYERS), ("status", STATUSES), ("kind", KINDS)):
        if _require(record, key, rule_id) not in allowed:
            raise RuleError(f"{rule_id}: unknown {key} {record[key]!r}")
    registers = tuple(_require(record, "scope", rule_id).get("registers") or ())
    if not registers:
        raise RuleError(f"{rule_id}: scope.registers is required — an unscoped rule is false somewhere")
    if unknown := set(registers) - REGISTERS:
        raise RuleError(f"{rule_id}: unknown registers {sorted(unknown)}")
    if rule_id.split(".")[0] not in STYLES:
        raise RuleError(f"{rule_id}: an id starts with its style, one of {STYLES}")
    source = _require(record, "source", rule_id)
    if not source.get("work") or not (source.get("page") or source.get("evidence")):
        raise RuleError(f"{rule_id}: source.work and source.page or source.evidence are required")
    academy = record.get("academy")
    if rule_id.startswith("ghazal.") and (not academy or academy.get("position") not in ACADEMY_POSITIONS):
        raise RuleError(f"{rule_id}: academy.position is required, one of {sorted(ACADEMY_POSITIONS)}")
    for key in ("title", "statement"):
        if not _require(record, key, rule_id).get("fa"):
            raise RuleError(f"{rule_id}: {key}.fa is required — the Persian is authoritative")
    examples = _require(record, "examples", rule_id)
    ok = tuple(examples.get("ok") or ())
    bad = tuple(Example(v["text"], v.get("expect")) if isinstance(v, dict) else Example(v)
                for v in examples.get("violations") or ())
    if not ok or not bad:
        raise RuleError(f"{rule_id}: both examples.ok and examples.violations are required")
    spec = record.get("detector")
    return Rule(
        rule_id, record["layer"], record["title"], record["statement"], record["kind"], record["status"],
        registers, Source(**{k: str(v) if k == "page" else v for k, v in source.items()}), ok, bad,
        _detector(spec, rule_id, record["layer"]) if spec else None,
        record.get("severity", "violation"), record.get("confidence", "high"), record.get("notes"),
        tuple(record.get("disputes") or ()), academy,
    )


def load_rules(paths: Sequence[Path] | None = None) -> tuple[list[Rule], dict[str, tuple[str, ...]]]:
    """Every rule in the packs, rejecting an id used twice, and the rules each
    style adopts from another."""
    rules, seen, adopts = [], {}, {}
    for path in sorted(paths) if paths is not None else sorted(RULES.glob("*.toml")):
        pack = tomllib.loads(path.read_text(encoding="utf-8"))
        if pack.get("adopts"):
            adopts[path.stem.split("-")[0]] = tuple(pack["adopts"])
        for record in pack.get("rule", []):
            rule = parse_rule(record)
            if rule.id in seen:
                raise RuleError(f"duplicate rule id {rule.id!r} in {path} and {seen[rule.id]}")
            seen[rule.id] = path
            rules.append(rule)
    if missing := {a for ids in adopts.values() for a in ids} - seen.keys():
        raise RuleError(f"adopted rules not found: {sorted(missing)}")
    return rules, adopts


@dataclass(frozen=True, slots=True)
class Check:
    findings: tuple[Finding, ...]
    abstentions: tuple[Abstention, ...]
    rules_run: int
    registers: tuple[str, ...]
    rulebase_version: str = RULEBASE_VERSION


@dataclass
class RuleBase:
    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def load(cls, paths: Sequence[Path] | None = None, *, style: str | None = None, verify: bool = True) -> RuleBase:
        """All rules, or one style's: its own and the ones it adopts."""
        rules, adopts = load_rules(paths)
        if style is not None:
            if style not in STYLES:
                raise ValueError(f"unknown style {style!r}, one of {STYLES}")
            rules = [r for r in rules if r.style == style or r.id in adopts.get(style, ())]
        base = cls(rules)
        if verify:
            base.verify()
        return base

    def verify(self) -> None:
        for rule in self.rules:
            rule.verify()

    def by_id(self, rule_id: str) -> Rule:
        return {r.id: r for r in self.rules}[rule_id]

    def for_registers(self, registers: Iterable[str]) -> list[Rule]:
        return [r for r in self.rules if r.runs and r.applies_in(registers)]

    def summary(self) -> dict:
        """``recorded`` minus ``executable`` is what is written down and cannot
        yet be checked."""
        return {
            "rulebase_version": RULEBASE_VERSION,
            "recorded": len(self.rules),
            "executable": sum(r.executable for r in self.rules),
            "running": sum(r.runs for r in self.rules),
            "by_layer": dict(Counter(r.layer for r in self.rules)),
            "by_status": dict(Counter(r.status for r in self.rules)),
            "by_kind": dict(Counter(r.kind for r in self.rules)),
            "sources": sorted({r.source.work for r in self.rules}),
        }

    def check(self, text: str, *, registers: Sequence[str] = ("formal_written",),
              min_confidence: str = "medium") -> Check:
        if min_confidence not in CONFIDENCE:
            raise ValueError(f"min_confidence is one of {sorted(CONFIDENCE)}")
        if unknown := set(registers) - REGISTERS:
            raise ValueError(f"unknown registers {sorted(unknown)}")
        floor = CONFIDENCE[min_confidence]
        rules = self.for_registers(registers)
        findings, abstentions = [], []
        for rule in rules:
            got, skipped = rule.scan(text)
            findings += [f for f in got if CONFIDENCE.get(f.confidence, 0) >= floor]
            abstentions += skipped
        rank = {"violation": 0, "advisory": 1, "house": 2}
        findings.sort(key=lambda f: (f.span, f.suggestion or "", rank.get(f.severity, 3)))
        findings = [f for i, f in enumerate(findings)
                    if not i or (f.span, f.suggestion) != (findings[i - 1].span, findings[i - 1].suggestion)]
        findings.sort(key=lambda f: (f.span[0], f.rule_id))
        abstentions.sort(key=lambda a: (a.span[0], a.rule_id))
        return Check(tuple(findings), tuple(abstentions), len(rules), tuple(registers))

    def profile(self, texts: Iterable[str], *, registers: Sequence[str] = ("formal_written",)) -> dict:
        """Coverage over a document set. The abstention histogram names the next
        lexicon to build."""
        documents = with_finding = 0
        by_rule, by_kind, by_reason = Counter(), Counter(), Counter()
        for text in texts:
            documents += 1
            analysis = self.check(text, registers=registers)
            with_finding += bool(analysis.findings)
            by_rule.update(f.rule_id for f in analysis.findings)
            by_kind.update(f.kind for f in analysis.findings)
            by_reason.update(a.reason for a in analysis.abstentions)
        return {
            "rulebase_version": RULEBASE_VERSION,
            "registers": list(registers),
            "documents": documents,
            "documents_with_finding": with_finding,
            "document_coverage": round(with_finding / documents, 4) if documents else 0.0,
            "findings_by_rule": dict(by_rule.most_common()),
            "findings_by_kind": dict(by_kind.most_common()),
            "abstentions_by_reason": dict(by_reason.most_common()),
        }


@lru_cache(maxsize=None)
def rulebase(style: str = "ghazal") -> RuleBase:
    """One style's rules, verified once."""
    return RuleBase.load(style=style)
