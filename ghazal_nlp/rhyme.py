"""قافیه و ردیف: a poem's rhyme and refrain.

Which hemistichs rhyme depends on the form: in a ghazal or qasida the first
couplet's two and every second one after, in a qeta every second one, in a
masnavi each couplet's own pair, and in a robāʿī or dobeyti lines 1, 2 and 4.
The radif is the words they all end with; the rhyme letters are the letters
before it that they share. The even hemistichs decide the rhyme, and the opening
one is checked against it, since a robāʿī's first line may not rhyme. Matching is
on letters only: harakat, ZWNJ and punctuation don't count, آ is ا, and an
ezāfe written هٔ is هی, as it's read.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .text import fold

FORMS = ("ghazal", "qasida", "qeta", "masnavi", "robai", "dobeyti")
LETTERS = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ی", "ۀ": "هی", "ة": "ه"})
# A line may miss the rhyme the others share: one in ten, at most.
TOLERANCE = 0.1
PUNCTUATION = "،؛؟!?.,:;«»\"'()[]…-–—"


@dataclass(frozen=True, slots=True)
class Rhyme:
    lines: tuple[int, ...]
    radif: str = ""
    qafiya: str = ""
    words: tuple[str, ...] = ()
    off: tuple[int, ...] = ()
    repeated: tuple[str, ...] = ()
    abstention: str | None = None

    @property
    def rawi(self) -> str:
        """The last rhyme letter: the rawi, unless the rhyme ends in a suffix (دل‌ها)."""
        return self.qafiya[-1:]

    @property
    def letters(self) -> str:
        """The rhyme letters and the radif without spaces, as Ganjoor writes them."""
        return self.qafiya + key(self.radif.replace(" ", ""))


def key(word: str) -> str:
    word = fold(word).replace("هٔ", "هی").translate(LETTERS)
    return "".join(c for c in word if unicodedata.category(c).startswith("L"))


def words(line: str) -> list[tuple[str, str]]:
    """(as written, letters) for each word of ``line``."""
    return [(w.strip(PUNCTUATION), k) for w in fold(line).split() if (k := key(w))]


def groups(count: int, form: str) -> list[tuple[int, ...]] | None:
    """The hemistichs that rhyme with each other, by index."""
    if form in ("ghazal", "qasida"):
        return [(0, *range(1, count, 2))]
    if form == "qeta":
        return [tuple(range(1, count, 2))]
    if form == "masnavi":
        return [(i, i + 1) for i in range(0, count - 1, 2)]
    if form in ("robai", "dobeyti"):
        return [tuple(i for i in (0, 1, 3) if i < count)]
    return None


def _shared(tails: list[str], need: int) -> str:
    """The longest ending at least ``need`` of ``tails`` have."""
    for size in range(max(map(len, tails)), 0, -1):
        counts: dict[str, int] = {}
        for t in tails:
            if len(t) >= size:
                counts[t[-size:]] = counts.get(t[-size:], 0) + 1
        if counts and max(counts.values()) >= need:
            return max(counts, key=lambda k: (counts[k], k))
    return ""


def rhyme_of(lines: list[str], indexes: tuple[int, ...], tolerance: float = TOLERANCE, optional: int = 0) -> Rhyme:
    """The rhyme ``indexes`` share; the first ``optional`` of them may rhyme or not
    (a robāʿī's first line), so they're checked but don't decide it."""
    written = [words(lines[i]) for i in indexes]
    every = [[k for _, k in w] for w in written]
    optional = optional if len(every) - optional >= 2 else 0
    ends = every[optional:]
    if len(ends) < 2 or not all(every):
        return Rhyme(indexes, abstention="too_few_lines")
    need = len(ends) - int(len(ends) * tolerance)
    letters = _shared(["".join(e) for e in ends], need)
    radif: list[str] = []
    while True:
        at = [e[-len(radif) - 1] for e in ends if len(e) > len(radif)]
        candidate = [_shared(at, 0) if not at else max(set(at), key=at.count), *radif]
        joined = "".join(candidate)
        if sum(e[-len(candidate):] == candidate for e in ends) < need or not letters.endswith(joined) \
                or len(letters) <= len(joined):
            break
        radif = candidate
    qafiya = letters[:len(letters) - len("".join(radif))]
    if not qafiya:
        return Rhyme(indexes, abstention="no_shared_rhyme")
    rhyme_words = tuple(w[-len(radif) - 1][0] if len(w) > len(radif) else "" for w in written)
    off = tuple(i for i, e in zip(indexes, every) if not "".join(e).endswith(letters))
    seen, repeated = {}, []
    for w in rhyme_words:
        k = key(w)
        if k in seen and seen[k] not in repeated:
            repeated.append(seen[k])
        seen.setdefault(k, w)
    radif_written = next((" ".join(x for x, _ in w[-len(radif):]) for w in written[optional:]
                          if [k for _, k in w[-len(radif):]] == radif), "") if radif else ""
    return Rhyme(indexes, radif_written, qafiya, rhyme_words, off, tuple(repeated))


def scan(lines: list[str], form: str, tolerance: float = TOLERANCE) -> list[Rhyme]:
    """One Rhyme per group of rhyming hemistichs: one for most forms, one per
    couplet in a masnavi. An unknown form abstains."""
    lines = [line for line in lines if line.strip()]
    found = groups(len(lines), form)
    if found is None:
        return [Rhyme((), abstention="form_unknown")]
    optional = 1 if form in ("ghazal", "qasida", "robai", "dobeyti") else 0
    return [rhyme_of(lines, g, tolerance, optional) for g in found]
