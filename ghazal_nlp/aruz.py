"""عروض: a deterministic scanner for classical Persian metre.

A written hemistich admits a set of syllabifications, because Persian does not
write short vowels, the ezāfe or gemination. A metre is admitted when some
syllabification realises its weight pattern. The scanner reports one metre,
several (the writing is ambiguous), or an abstention with a reason, and names
the licences each reading needed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

SHORT, LONG = "u", "-"
DATA = Path(__file__).parent / "data"

# Each foot scans as its own pronunciation. Ganjoor drops the final damma of the
# مکفوف and مقبوض feet, so مفعولُ appears as مفعول.
FEET = {
    "فاعلاتن": "-u--", "فاعلات": "-u-u", "فاعلن": "-u-", "فاعلُ": "-uu",
    "فعلاتن": "uu--", "فعلات": "uu-u", "فعلن": "uu-", "فعل": "u-",
    "فعولن": "u--", "مفاعیلن": "u---", "مفاعیل": "u--u", "مفاعلن": "u-u-",
    "مفاعلتن": "u-uu-", "مفعولن": "---", "مفعول": "--u", "مفتعلن": "-uu-",
    "مفتعلات": "-uu-u", "مستفعلن": "--u-", "مستفعلتن": "--uu-", "مستفعل": "--uu",
    "متفاعلن": "uu-u-", "متفاعلتن": "uu-uu-", "متفاعلاتن": "uu-u--", "متفعلن": "u-u-",
    "فع": "-", "لن": "-", "فع‌لن": "--",
}

RUBAI = "وزن رباعی"


def pattern_for_label(label: str) -> str | None:
    """A Ganjoor label's arkān expanded into weights, or None if a foot is unknown."""
    feet = [FEET.get(foot) for foot in label.split("(")[0].split()]
    return "".join(feet) if feet and None not in feet else None


# --- letters ------------------------------------------------------------------

FOLD = str.maketrans("يىﻯكةۀئؤأإٱ", "یییکههءءءاا")
STRIP = re.compile("[ً-ٰ٘ۖ-ۭـ‌-‏﻿]")
WORD = re.compile(r"[^\s .,،؛؟!:;«»\"'()\[\]{}\-–—ـ/\\|*_+=]+")
CONSONANTS = set("بپتثجچحخدذرزژسشصضطظعغفقکگلمنهویء")
LETTERS = CONSONANTS | {"ا", "آ"}

FATHA, DAMMA, KASRA, SUKUN, SHADDA, HAMZA = "َ", "ُ", "ِ", "ْ", "ّ", "ٔ"
ZWNJ = "‌"
VOWELS = FATHA + DAMMA + KASRA
READABLE = frozenset(VOWELS + SUKUN + SHADDA + HAMZA)

DEVICES = (
    "long_vowel_shortening", "hamze_elision", "ezafe_glide", "tashdid",
    "silent_vav_after_khe", "final_syllable_free", "overlong_as_long",
    "conjunction_vav", "ebdal", "initial_long", "ezafe", "written_gemination",
    "ezafe_lengthening", "written_ezafe", "ast_contraction",
)
(SHORTEN, ELIDE, GLIDE, TASHDID, XVA, FINAL, OVERLONG, VAV, EBDAL, INITIAL,
 EZAFE, WRITTEN_GEMINATION, EZAFE_LONG, WRITTEN_EZAFE, AST) = (1 << i for i in range(15))

# Uniform cost per licence. The free ones are ordinary phonology, or the
# editor's own marks, not licences.
FREE = FINAL | ELIDE | VAV | XVA | WRITTEN_GEMINATION | WRITTEN_EZAFE
COST = [(mask & ~FREE).bit_count() for mask in range(1 << 15)]


def device_names(mask: int) -> tuple[str, ...]:
    return tuple(name for i, name in enumerate(DEVICES) if mask >> i & 1)


@dataclass(frozen=True, slots=True)
class Skeleton:
    """A hemistich as the letters that can bear a syllable, word by word."""

    chars: tuple[str, ...]
    word: tuple[int, ...]
    initial: tuple[bool, ...]
    final: tuple[bool, ...]
    words: tuple[str, ...]
    marks: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.chars)


class ScriptError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Marks:
    """Which editorial marks are read as constraints.

    vowels: a written vowel makes its letter an onset. sukun: the letter opens
    no syllable. shadda: the letter closes one syllable and opens the next.
    ezafe_he: هٔ is a silent ه carrying an ezāfe. zwnj: a morpheme edge inside
    one phonological word. Marks only narrow, except that a shadda or a written
    ezāfe may license a gemination, and a ZWNJ restores word edges.
    """

    vowels: bool = True
    sukun: bool = True
    shadda: bool = True
    ezafe_he: bool = True
    zwnj: bool = True


def _kept_marks(raw: str, ch: str, initial: bool, final: bool, word: str, sel: Marks) -> str:
    if not raw or ch in "اآ":
        return ""
    out = ""
    vowels = "".join(m for m in raw if m in VOWELS)
    if ch == "ه" and final and HAMZA in raw and sel.ezafe_he:
        out += HAMZA
    if word == "و":
        vowels = vowels.replace(DAMMA, "")  # و with a damma is /o/, which has no onset
    if sel.vowels and vowels:
        out += vowels[0]
    if sel.sukun and SUKUN in raw:
        out += SUKUN
    if sel.shadda and SHADDA in raw and not initial:
        out += SHADDA
    return out


def prepare(line: str, *, marks: Marks | None = None) -> Skeleton:
    """Fold, strip and split a line; with ``marks``, keep them beside the letters."""
    text = unicodedata.normalize("NFC", line)
    if marks is not None:
        text = text.replace("ۀ", "ه" + HAMZA)
    text = text.translate(FOLD)
    kept: list[str] = []
    on: list[str] = []
    for ch in text:
        if STRIP.fullmatch(ch):
            if marks is not None and kept and (ch in READABLE or (ch == ZWNJ and marks.zwnj)):
                on[-1] += ch
            continue
        kept.append(ch)
        on.append("")
    text = "".join(kept)

    chars, ids, initial, final, letter_marks, words = [], [], [], [], [], []
    for match in WORD.finditer(text):
        word = match.group()
        if any(c not in LETTERS for c in word):
            if any(c.isdigit() or c.isascii() for c in word):
                raise ScriptError("non_persian_script")
            raise ScriptError("unknown_grapheme")
        joined = False
        for k, ch in enumerate(word):
            raw = on[match.start() + k] if marks is not None else ""
            first = k == 0 or joined
            last = k == len(word) - 1 or (ZWNJ in raw and ch == "ه")
            joined = ZWNJ in raw
            chars.append(ch)
            ids.append(len(words))
            initial.append(first)
            final.append(last)
            if marks is not None:
                letter_marks.append(_kept_marks(raw, ch, first, last, word, marks))
        words.append(word)
    if not chars:
        raise ScriptError("empty_line")
    return Skeleton(tuple(chars), tuple(ids), tuple(initial), tuple(final), tuple(words),
                    tuple(letter_marks) if any(letter_marks) else ())


# --- syllables ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Devices:
    """The licences the scanner may apply. One switched off makes the lines that
    need it abstain."""

    hamze_elision: bool = True
    conjunction_vav: bool = True
    ezafe_glide: bool = True
    silent_vav_after_khe: bool = True
    overlong_as_long: bool = True
    ebdal: bool = True
    initial_long: bool = True
    tashdid: bool = True
    ezafe_long: bool = True
    tashdid_before_vowel: bool = True
    ast_contraction: bool = True
    ebdal_first_foot: bool = True
    caesura: bool = True


WEIGHT = {
    (SHORT, 0): SHORT, (SHORT, 1): LONG, (SHORT, 2): LONG + SHORT,
    (LONG, 0): LONG, (LONG, 1): LONG + SHORT, (LONG, 2): LONG + SHORT,
}


def _is_coda(skel: Skeleton, k: int, word: int) -> bool:
    return k < len(skel) and skel.word[k] == word and skel.chars[k] in CONSONANTS


def _onsets(skel: Skeleton, i: int):
    """(nucleus start, forced nucleus, word) for each onset at ``i``."""
    ch = skel.chars[i]
    if ch == "آ":
        yield i + 1, LONG, skel.word[i]
    elif ch == "ا":
        # The glottal onset word-initially, or a hiatus after another vowel letter.
        if skel.initial[i] or (i and skel.chars[i - 1] in "اویآ"):
            yield i + 1, None, skel.word[i]
    elif ch == "ه" and skel.final[i] and not (skel.marks and any(v in skel.marks[i] for v in VOWELS)):
        # A final ه opens a syllable only as the /h/ after ا (شاه یمن is šâ-he).
        if i and skel.chars[i - 1] in "اآ" and skel.word[i - 1] == skel.word[i] and not (
            skel.marks and skel.marks[i]
        ):
            yield i + 1, None, skel.word[i]
    elif ch in CONSONANTS:
        yield i + 1, None, skel.word[i]


def _nuclei(skel: Skeleton, k: int, word: int):
    """(kind, next index, devices) for the nucleus at ``k``."""
    yield SHORT, k, 0
    if k >= len(skel) or skel.word[k] != word:
        return
    ch = skel.chars[k]
    if ch == "آ" or (ch == "ا" and not skel.initial[k]):
        yield LONG, k + 1, 0
    elif ch in "وی":
        yield LONG, k + 1, 0
        yield SHORT, k + 1, SHORTEN
    elif ch == "ه" and skel.final[k]:
        yield SHORT, k + 1, 0


def _syllables(skel: Skeleton, i: int, devices: Devices):
    """(weights, next index, devices) for each syllable the letters admit at ``i``."""
    n = len(skel)
    if i >= n:
        return
    ch = skel.chars[i]
    if devices.silent_vav_after_khe and ch == "خ" and _khe_opens_stem(skel, i) and i + 1 < n and skel.chars[i + 1] == "و":
        yield from _from_onset(skel, i + 2, skel.word[i], devices, XVA)
    for start, forced, word in _onsets(skel, i):
        if forced is not None:
            yield from _codas(skel, forced, start, word, 0, devices)
        else:
            yield from _from_onset(skel, start, word, devices, 0)
    if devices.hamze_elision and ch in CONSONANTS and i + 1 < n:
        j = i + 1
        if skel.initial[j] and skel.chars[j] in "اآ":
            word = skel.word[j]
            if skel.chars[j] == "آ":
                yield from _codas(skel, LONG, j + 1, word, ELIDE, devices)
            else:
                if j + 1 < n and skel.word[j + 1] == word and skel.chars[j + 1] in "وی":
                    yield from _codas(skel, LONG, j + 2, word, ELIDE, devices)
                yield from _codas(skel, SHORT, j + 1, word, ELIDE, devices)
    if (devices.conjunction_vav and ch in CONSONANTS and i + 1 < n and skel.chars[i + 1] == "و"
            and skel.initial[i + 1] and skel.final[i + 1]):
        word = skel.word[i + 1]
        yield from _codas(skel, SHORT, i + 2, word, VAV, devices)
        yield from _codas(skel, LONG, i + 2, word, VAV, devices)
    if (devices.ast_contraction and ch in CONSONANTS and i + 2 < n and skel.chars[i + 1] == "ه"
            and skel.final[i + 1] and skel.word[i + 1] == skel.word[i]):
        start = _ast_start(skel, i + 2, whole=False)
        if start is not None:
            yield from _codas(skel, SHORT, start, skel.word[i + 2], AST, devices)


def _ast_start(skel: Skeleton, j: int, *, whole: bool) -> int | None:
    """Where the -st of an است (or ست) written at ``j`` begins."""
    if j >= len(skel) or not skel.initial[j]:
        return None
    start = j + 1 if skel.chars[j] == "ا" else j
    if "".join(skel.chars[start:start + 2]) != "ست" or skel.word[start + 1] != skel.word[j]:
        return None
    if whole and not skel.final[start + 1]:
        return None
    return start


STEM_PREFIXES = frozenset({"ب", "ن", "می", "نمی", "همی", "بی"})


def _khe_opens_stem(skel: Skeleton, i: int) -> bool:
    if skel.initial[i] or (i + 2 < len(skel) and skel.word[i + 2] == skel.word[i] and skel.chars[i + 2] == "ا"):
        return True
    start = i
    while start and skel.word[start - 1] == skel.word[i] and not skel.initial[start]:
        start -= 1
    return "".join(skel.chars[start:i]) in STEM_PREFIXES


@dataclass(frozen=True, slots=True)
class _MarkRules:
    onset: tuple[bool, ...]
    closed: tuple[bool, ...]
    geminate: tuple[bool, ...]
    ezafe_he: tuple[bool, ...]

    def moves(self, i: int, weights: str, nxt: int, dev: int, devices: Devices):
        n = len(self.onset)
        if self.closed[i]:
            return
        if any(self.onset[j] or self.geminate[j] for j in range(i + 1, min(nxt, n))):
            return
        if nxt < n and self.geminate[nxt] and not dev & (WRITTEN_GEMINATION | TASHDID):
            return
        he = next((j for j in range(i, min(nxt, n)) if self.ezafe_he[j]), None)
        if he is None:
            yield weights, nxt, dev
            return
        if nxt != he + 1 or weights != SHORT:
            return
        if nxt == n:
            yield weights, nxt, dev
            return
        # The ezāfe of هٔ is an extra short, the same syllable lengthened, or
        # merged into the silent vowel; never absent.
        yield SHORT + SHORT, nxt, dev | WRITTEN_EZAFE
        if devices.ezafe_long:
            yield SHORT + LONG, nxt, dev | WRITTEN_EZAFE | EZAFE_LONG
            yield LONG, nxt, dev | WRITTEN_EZAFE | EZAFE_LONG

    def glide(self, i: int) -> bool:
        return not (self.ezafe_he[i - 1] or self.closed[i - 1] or self.onset[i - 1])


@lru_cache(maxsize=4096)
def _mark_rules(skel: Skeleton) -> _MarkRules | None:
    if not skel.marks:
        return None
    onset = tuple(any(v in m for v in VOWELS) for m in skel.marks)
    return _MarkRules(
        onset,
        tuple(SUKUN in m for m in skel.marks),
        tuple(SHADDA in m and (onset[k] or not skel.final[k]) for k, m in enumerate(skel.marks)),
        tuple(HAMZA in m for m in skel.marks),
    )


def _moves(skel: Skeleton, i: int, devices: Devices):
    """The syllables at ``i`` that the letters and the marks admit."""
    rules = _mark_rules(skel)
    for move in _syllables(skel, i, devices):
        if rules is None:
            yield move
        else:
            yield from rules.moves(i, *move, devices)


def _glide_weights(devices: Devices) -> tuple[tuple[str, int], ...]:
    if devices.ezafe_long:
        return (SHORT, GLIDE), (LONG, GLIDE | EZAFE_LONG)
    return ((SHORT, GLIDE),)


def _glide(skel: Skeleton, i: int, devices: Devices) -> bool:
    """May an unwritten ezāfe glide (خانهٔ من, xâ-ne-ye) sit before letter ``i``?"""
    if not (devices.ezafe_glide and 0 < i < len(skel) and skel.final[i - 1] and skel.chars[i - 1] in "اویه"):
        return False
    rules = _mark_rules(skel)
    return rules is None or rules.glide(i)


def _from_onset(skel: Skeleton, start: int, word: int, devices: Devices, dev: int):
    for kind, after, ndev in _nuclei(skel, start, word):
        yield from _codas(skel, kind, after, word, dev | ndev, devices, bare=after == start)


def _codas(skel: Skeleton, kind: str, k: int, word: int, dev: int, devices: Devices, bare: bool = False):
    """Every (weight, next index, devices) the coda choices allow.

    An overlong syllable is canonically long + short, but many lines scan only
    if it fills one long, so both are offered and the second is a device.
    """
    n = len(skel)
    # A word cannot end in an unwritten short vowel; one that does carries an ezāfe.
    open_final = bare and (k >= n or skel.word[k] != word)
    ezafe = 0
    if open_final:
        ezafe = WRITTEN_EZAFE if skel.marks and any(v in skel.marks[k - 1] for v in VOWELS) else EZAFE
    yield WEIGHT[kind, 0], k, dev | ezafe
    if open_final and devices.ezafe_long:
        yield LONG, k, dev | ezafe | EZAFE_LONG
    for count in (1, 2):
        if not _is_coda(skel, k + count - 1, word):
            break
        weight = WEIGHT[kind, count]
        yield weight, k + count, dev
        if len(weight) == 2 and devices.overlong_as_long:
            yield LONG, k + count, dev | OVERLONG
    if devices.ast_contraction and kind == LONG:
        for count in (0, 1):
            end = k + count
            if count and not _is_coda(skel, k, word):
                break
            if end < n and skel.word[end] != word:
                start = _ast_start(skel, end, whole=True)
                if start is not None:
                    yield LONG + SHORT, start + 2, dev | AST
                    if devices.overlong_as_long:
                        yield LONG, start + 2, dev | AST | OVERLONG
    # Unwritten gemination: one letter closes a syllable and opens the next.
    if not _is_coda(skel, k, word):
        return
    if skel.marks and SHADDA in skel.marks[k]:
        yield WEIGHT[kind, 1], k, dev | WRITTEN_GEMINATION
    elif devices.tashdid and (
        _is_coda(skel, k + 1, word)
        or (devices.tashdid_before_vowel and k + 1 < n and skel.word[k + 1] == word and skel.chars[k + 1] in "اوی")
        or (skel.marks and skel.final[k] and any(v in skel.marks[k] for v in VOWELS))
    ):
        yield WEIGHT[kind, 1], k, dev | TASHDID


# --- patterns -----------------------------------------------------------------

CAESURA = "|"

# Doubled metres whose poets let words cross the half: دوری by their arkān only.
NOT_DOWRI = frozenset({
    "مفاعلن فعلاتن مفاعلن فعلاتن (مجتث مثمن مخبون)",
    "مفاعیل فعولن مفاعیل فعولن",
    "مفتعلن مفتعلن فع مفتعلن مفتعلن فع",
    "مفاعلن فع مفاعلن فع",
    "فعلاتن فعلن فعلاتن فعلن",
    "فعلات فعلن فعلات فعلن",
    "مستفعلن فعلاتن مستفعلن فعلاتن",
    "مفاعلن فعولن مفاعلن فعولن",
    "فاعلات فعلن فاعلات فعلن",
})


def dowri_half(label: str) -> int | None:
    """Where a دوری metre (two identical halves of mixed feet) divides."""
    feet = label.split("(")[0].split()
    k, odd = divmod(len(feet), 2)
    if odd or k < 2 or feet[:k] != feet[k:] or len(set(feet[:k])) == 1 or label.strip() in NOT_DOWRI:
        return None
    pattern = pattern_for_label(label)
    return None if pattern is None else len(pattern) // 2


class Trie:
    """Prefix trie over weight patterns, with a zero-width caesura edge in دوری metres."""

    __slots__ = ("children", "terminal", "depth", "patterns", "caesura")

    def __init__(self, patterns, caesuras=None):
        self.children: list[dict[str, int]] = [{}]
        self.terminal: list[int | None] = [None]
        self.depth = [0]
        self.patterns = list(patterns)
        self.caesura = bool(caesuras)
        for index, pattern in enumerate(self.patterns):
            half = caesuras.get(pattern) if caesuras else None
            if half is not None:
                pattern = pattern[:half] + CAESURA + pattern[half:]
            node = 0
            for ch in pattern:
                nxt = self.children[node].get(ch)
                if nxt is None:
                    nxt = len(self.children)
                    self.children.append({})
                    self.terminal.append(None)
                    self.depth.append(self.depth[node] + (ch != CAESURA))
                    self.children[node][ch] = nxt
                node = nxt
            self.terminal[node] = index

    def walk(self, node: int, weights: str) -> int | None:
        for ch in weights:
            node = self.children[node].get(ch, -1)
            if node < 0:
                return None
        return node

    def spill(self, node: int, weights: str) -> int | None:
        """An overlong ending the دوری half, its short carried across."""
        return self.walk(node, LONG + CAESURA + SHORT) if weights == LONG + SHORT else None


def _pattern_slots(weights: str, i: int, devices: Devices):
    """The positions one syllable may fill: as itself, by ابدال, or as a long opening."""
    yield weights, 0
    if devices.ebdal and weights == LONG:
        yield SHORT + SHORT, EBDAL
    if devices.initial_long and i == 0 and weights.startswith(LONG):
        yield SHORT + weights[1:], INITIAL


def _slot_blocked(licence: int, depth: int, devices: Devices) -> bool:
    return licence == EBDAL and depth < 2 and not devices.ebdal_first_foot


# --- the weight lexicon -------------------------------------------------------

PINNED, UNDERDETERMINED, VARIABLE, CONFLICT = "pinned", "underdetermined", "variable", "conflict"


def _word_spans(skel: Skeleton) -> list[tuple[int, int]]:
    spans, start = [], 0
    for i in range(1, len(skel) + 1):
        if i == len(skel) or skel.word[i] != skel.word[i - 1]:
            spans.append((start, i))
            start = i
    return spans


def _constrainable_words(skel: Skeleton, devices: Devices) -> list[bool]:
    """Words not touched by hamze elision or the conjunction و, which move letters
    across the word edge."""
    words = skel.words
    ok = [True] * len(words)
    for w in range(len(words) - 1):
        if not words[w] or not words[w + 1] or words[w][-1] not in CONSONANTS:
            continue
        nxt = words[w + 1]
        if (devices.hamze_elision and nxt[0] in "اآ") or (devices.conjunction_vav and nxt == "و"):
            ok[w] = ok[w + 1] = False
    return ok


@lru_cache(maxsize=100_000)
def _word_cuts(word: str, allowed: frozenset[str], devices: Devices):
    """(offset, weight, length) of syllables on a reading of ``word`` whose weight
    string is allowed, or None when the letters admit none."""
    skel = prepare(word)
    n = len(skel)
    prefixes = {p[:k] for p in allowed for k in range(len(p) + 1)}
    forward: list[set[str]] = [set() for _ in range(n + 1)]
    forward[0].add("")
    edges: list[list[tuple[str, str, int, str]]] = [[] for _ in range(n)]
    for i in range(n):
        if not forward[i]:
            continue
        for weights, nxt, _ in _syllables(skel, i, devices):
            if nxt > n:
                continue
            for prefix in forward[i]:
                grown = prefix + weights
                if grown in prefixes:
                    edges[i].append((prefix, grown, nxt, weights))
                    forward[nxt].add(grown)
    complete: list[set[str]] = [set() for _ in range(n + 1)]
    complete[n] = {p for p in forward[n] if p in allowed}
    for i in range(n - 1, -1, -1):
        for prefix, grown, nxt, _ in edges[i]:
            if grown in complete[nxt]:
                complete[i].add(prefix)
    if "" not in complete[0]:
        return None
    return frozenset(
        (i, weights, nxt - i)
        for i in range(n)
        for prefix, grown, nxt, weights in edges[i]
        if prefix in complete[i] and grown in complete[nxt]
    )


@dataclass(frozen=True, slots=True)
class WeightEntry:
    word: str
    status: str
    weights: tuple[str, ...]
    occurrences: int
    agreement: float
    support: tuple[tuple[str, int], ...]


class WeightLexicon:
    """Word → the weight strings it took under its poem's metre label.

    Supervised: derived from Ganjoor's labels on the development split, so a
    result that uses it is never unsupervised scansion.
    """

    def __init__(self, entries, provenance=None, *, min_occurrences: int = 2, max_weights: int = 3):
        self.entries = tuple(entries)
        self.provenance = dict(provenance or {})
        self.allowed = {
            e.word: frozenset(e.weights)
            for e in self.entries
            if e.status != CONFLICT and e.weights and e.occurrences >= min_occurrences and len(e.weights) <= max_weights
        }

    def __len__(self) -> int:
        return len(self.allowed)

    def summary(self) -> dict:
        counts = Counter(e.status for e in self.entries)
        return {"words": len(self.entries), "constraining": len(self.allowed),
                **{s: counts[s] for s in (PINNED, UNDERDETERMINED, VARIABLE, CONFLICT)}, **self.provenance}

    @classmethod
    def load(cls, path: Path = DATA / "weights.tsv", **kwargs) -> WeightLexicon:
        provenance, entries = {}, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                key, sep, value = line[1:].strip().partition(":")
                if sep and " " not in key:
                    provenance[key] = value.strip()
            elif line.strip():
                word, status, weights, occurrences, agreement, support = line.split("\t")
                entries.append(WeightEntry(
                    word, status, tuple(w for w in weights.split("|") if w), int(occurrences), float(agreement),
                    tuple((w.rsplit(":", 1)[0], int(w.rsplit(":", 1)[1])) for w in support.split("|") if w),
                ))
        return cls(entries, provenance, **kwargs)


# --- the scanner --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scansion:
    """One hemistich: the cheapest metres, or an abstention. ``candidates`` holds
    every admitted metre with its licence cost; ``gated`` says the lexicon was used."""

    line: str
    metres: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    licences: int = 0
    candidates: tuple[tuple[str, int], ...] = ()
    abstention: str | None = None
    gated: bool = False


@dataclass(frozen=True, slots=True)
class MetreEntry:
    label: str
    pattern: str
    provenance: str
    poems: int


class Inventory:
    """Weight patterns and the labels that realise them."""

    def __init__(self, entries):
        self.entries = tuple(entries)
        self.pattern_labels: dict[str, list[str]] = {}
        self.label_patterns: dict[str, list[str]] = {}
        for e in self.entries:
            self.pattern_labels.setdefault(e.pattern, []).append(e.label)
            self.label_patterns.setdefault(e.label, []).append(e.pattern)
        # Patterns differing only in the first syllable are one metre; each
        # reports the label it most often carries.
        poems = {e.label: e.poems for e in self.entries}
        metres: dict[tuple[int, str], list[str]] = {}
        for pattern in self.pattern_labels:
            metres.setdefault((len(pattern), pattern[1:]), []).append(pattern)
        for group in metres.values():
            labels = {label for p in group for label in self.pattern_labels[p]}
            if len(labels) > 1:
                major = max(sorted(labels), key=poems.__getitem__)
                for p in group:
                    self.pattern_labels[p] = [major]
        self.patterns = tuple(sorted(self.pattern_labels))
        self.trie = Trie(self.patterns)
        halves = {e.pattern: dowri_half(e.label) for e in self.entries}
        self.caesuras = {p: h for p, h in halves.items() if h is not None}
        self.caesura_trie = Trie(self.patterns, self.caesuras)

    def __len__(self) -> int:
        return len(self.label_patterns)

    @classmethod
    def load(cls, path: Path = DATA / "metres.tsv") -> Inventory:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
        return cls(MetreEntry(label, pattern, source, int(poems)) for label, pattern, source, poems in rows)


@dataclass
class Scanner:
    inventory: Inventory = field(default_factory=Inventory.load)
    devices: Devices = Devices()
    lexicon: WeightLexicon | None = None
    marks: Marks | None = Marks()
    max_weights: int = 4

    @property
    def trie(self) -> Trie:
        return self.inventory.caesura_trie if self.devices.caesura else self.inventory.trie

    def scan(self, line: str) -> Scansion:
        try:
            skel = prepare(line, marks=self.marks)
        except ScriptError as e:
            return Scansion(line, abstention=e.reason)
        gate = self._gate(skel)
        hits = self._search(skel, gate)
        if not hits and gate is not None:  # the lexicon is a hypothesis; back off
            gate = None
            hits = self._search(skel, None)
        if not hits:
            return Scansion(line, abstention=self._why_not(skel))
        cheapest = min(cost for cost, _, _ in hits.values())
        labels, weights, patterns, candidates, mask = [], [], [], {}, 0
        for index in sorted(hits, key=lambda k: (hits[k][0], self.trie.patterns[k])):
            cost, realised, used = hits[index]
            pattern = self.trie.patterns[index]
            found = self.inventory.pattern_labels[pattern]
            for label in found:
                candidates.setdefault(label, cost)
            if cost == cheapest:
                patterns.append(pattern)
                weights.append(realised)
                mask |= used
                labels.extend(found)
        return Scansion(line, tuple(dict.fromkeys(labels)), tuple(patterns), tuple(weights[:self.max_weights]),
                        device_names(mask), cheapest, tuple(candidates.items()), gated=gate is not None)

    def _gate(self, skel: Skeleton):
        """Per letter, the (weight, next) syllables the lexicon allows, or None."""
        if self.lexicon is None:
            return None
        eligible = _constrainable_words(skel, self.devices)
        gate = [None] * len(skel)
        covered = False
        for wid, (start, end) in enumerate(_word_spans(skel)):
            # An editor's mark on the word outranks the lexicon.
            if not eligible[wid] or (skel.marks and any(skel.marks[start:end])):
                continue
            allowed = self.lexicon.allowed.get(skel.words[wid])
            cuts = allowed and _word_cuts(skel.words[wid], allowed, self.devices)
            if not cuts:
                continue
            by_offset: dict[int, set] = {}
            for offset, weights, length in cuts:
                by_offset.setdefault(offset, set()).add((weights, start + offset + length))
            for offset in range(end - start):
                gate[start + offset] = frozenset(by_offset.get(offset, ()))
            covered = True
        return gate if covered else None

    def _search(self, skel: Skeleton, gate) -> dict[int, tuple[int, str, int]]:
        """Pattern index → (licences, weights, devices) of its cheapest reading."""
        trie, devices, n = self.trie, self.devices, len(skel)
        frontier: list[dict[int, tuple[int, str, int]]] = [{} for _ in range(n + 1)]
        frontier[0][0] = (0, "", 0)
        hits: dict[int, tuple[int, str, int]] = {}

        def offer(store, node, path):
            prior = store.get(node)
            if prior is None or path[0] < prior[0]:
                store[node] = path

        for i in range(n):
            nodes = frontier[i]
            if not nodes:
                continue
            if _glide(skel, i, devices):
                for node, (cost, sofar, mask) in list(nodes.items()):
                    for weight, dev in _glide_weights(devices):
                        landed = trie.walk(node, weight)
                        if landed is not None:
                            offer(nodes, landed, (cost + COST[dev], sofar + weight, mask | dev))
            if trie.caesura and i and skel.word[i] != skel.word[i - 1]:
                for node, path in list(nodes.items()):
                    half = trie.children[node].get(CAESURA)
                    if half is not None:
                        offer(nodes, half, path)
            allowed = gate[i] if gate is not None else None
            for weights, nxt, dev in _moves(skel, i, devices):
                if allowed is not None and (weights, nxt) not in allowed:
                    continue
                for realised, licence in _pattern_slots(weights, i, devices):
                    extra = COST[dev] + COST[licence]
                    for node, (cost, sofar, mask) in nodes.items():
                        if licence and _slot_blocked(licence, trie.depth[node], devices):
                            continue
                        path = (cost + extra, sofar + realised, mask | dev | licence)
                        landed = trie.walk(node, realised)
                        if landed is not None:
                            offer(frontier[nxt], landed, path)
                        if nxt != n:
                            if trie.caesura and skel.word[nxt] != skel.word[nxt - 1]:
                                spilled = trie.spill(node, realised)
                                if spilled is not None:
                                    offer(frontier[nxt], spilled, path)
                            continue
                        # The last syllable of a hemistich counts long whatever it is.
                        for ch, child in trie.children[node].items():
                            if trie.terminal[child] is not None and child != landed:
                                offer(hits, trie.terminal[child], (path[0], sofar + ch, path[2] | FINAL))
        for node, path in frontier[n].items():
            if trie.terminal[node] is not None:
                offer(hits, trie.terminal[node], path)
        return hits

    def _why_not(self, skel: Skeleton) -> str:
        n = len(skel)
        reach: list[set[int]] = [set() for _ in range(n + 1)]
        reach[0].add(0)
        for i in range(n):
            if not reach[i]:
                continue
            if _glide(skel, i, self.devices):
                reach[i].update({c + 1 for c in reach[i]})
            for weights, nxt, _ in _moves(skel, i, self.devices):
                reach[nxt].update(c + len(weights) for c in reach[i])
        lengths = reach[n] | {c - 1 for c in reach[n] if c > 0}
        if not lengths:
            return "no_syllabification"
        if not lengths & {len(p) for p in self.inventory.patterns}:
            return "length_outside_inventory"
        return "weights_unmatched"


@lru_cache(maxsize=2)
def scanner(lexicon: bool = True) -> Scanner:
    """The shipped scanner; ``lexicon=False`` is the unsupervised one."""
    return Scanner(lexicon=WeightLexicon.load() if lexicon else None)


# --- poems --------------------------------------------------------------------

MAX_LINES = 40
VOTE_MARGIN = 6  # a line votes for every metre within this many licences of its cheapest


def hemistichs(text: str) -> list[str]:
    """One hemistich per line; lines under six letters carry no metre."""
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if sum(c in LETTERS for c in line) >= 6]


def votes(scansion: Scansion) -> list[tuple[str, int]]:
    ceiling = scansion.licences + VOTE_MARGIN
    return [(metre, cost) for metre, cost in scansion.candidates if cost <= ceiling]


def winners(votes: Counter, costs: Counter, hemistich_count: int) -> list[str]:
    """Most votes, then fewest licences; past a quatrain the robāʿī loses ties,
    since its many shapes almost always find a cheaper reading."""
    def rank(m):
        return votes[m], -(hemistich_count > 4 and RUBAI in m), -costs[m]

    best = max(rank(m) for m in votes)
    return sorted(m for m in votes if rank(m) == best)


@dataclass(frozen=True, slots=True)
class PoemScansion:
    metres: tuple[str, ...] = ()
    support: int = 0
    lines: int = 0
    abstained: int = 0
    licences: int = 0
    abstentions: tuple[tuple[str, int], ...] = ()


def scan_poem(text: str, scan: Scanner | None = None, *, max_lines: int = MAX_LINES) -> PoemScansion:
    """A poem's metre: the one most of its hemistichs admit."""
    scan = scan or scanner()
    tally, costs, reasons = Counter(), Counter(), Counter()
    scanned = 0
    lines = hemistichs(text)
    for line in lines[:max_lines]:
        s = scan.scan(line)
        if not s.metres:
            reasons[s.abstention] += 1
            continue
        scanned += 1
        for metre, cost in votes(s):
            tally[metre] += 1
            costs[metre] += cost
    if not tally:
        return PoemScansion(lines=scanned, abstained=reasons.total(), abstentions=tuple(reasons.most_common()))
    best = winners(tally, costs, len(lines))
    return PoemScansion(tuple(best), tally[best[0]], scanned, reasons.total(), costs[best[0]],
                        tuple(reasons.most_common()))


def as_dict(result) -> dict:
    return asdict(result)
