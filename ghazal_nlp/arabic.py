"""Arabic inside Persian, found conservatively: passages, and set phrases.

A passage is Arabic syntax in context, not Arabic script: a shared word, a name,
a ة or a mark alone decides nothing. Persian words and punctuation bound the
spans, and offsets index the original text. A phrase is a formula or set phrase
(علیه السلام، مع ذلک) or a preposition or vocative before an Arabic article
(فی الحال، ایها الامیر). No span is an abstention, not a verdict that the text
is Persian. Quran wording is a separate, exact path over copies the caller
supplies.

``words``, ``spans`` and ``mask`` use a silver-trained contextual model.
``ArabicDetector`` and ``detector`` retain the original conservative rules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"

MARKS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_LETTER = r"[ء-غف-يٮ-ٯٱ-ۓۺ-ۼۿ]"
_TAIL = r"[ء-غف-يٮ-ٯٱ-ۓۺ-ۼۿـً-ٰٟۖ-ۭ]*"
TOKEN = re.compile(_LETTER + _TAIL + r"(?:[‌‍][‌‍ً-ٰٟۖ-ۭ]*" + _LETTER + _TAIL + r")*")
FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "أ": "ا", "إ": "ا", "ٱ": "ا"})
PERSIAN = frozenset("را است این آن از که برای بود شد گفت کرده خود یک در نیست هستند"
                    " بودند باشد شود تا چون اگر مگر اما پس نیز چنین همچنین چرا چگونه"
                    " گفتند میگوید میگویند گوید گویند فرمود یعنی نوشته ترجمه دارد داشت"
                    " کردن کردند شده میشود میباشد درباره بسیار هرچه هرکه ایشان آنکه"
                    " هر با او وی تو مرا اوست کنم کنند نکرد میداشتند میرانند میگویند"
                    " آریم فراهم ازیشان".split())
# Context features, not a claim that each form is exclusive to Arabic.
STRONG = frozenset("الذین الذی التی هذا هذه ذلک اولئک لیس لن لم لقد الی انما فانما فان"
                   " فقال فقالوا قالوا یقول یقولون کانت کانوا اذ اذا ثم حیث لکن لعل"
                   " لعلکم انکم انهم انه انی انا انک مما ممن کیف ایها ایتها".split())
WEAK = frozenset("ان فی عن قال علی من ما هو هی هم نحن انت لا الا کل کان مع بین عند"
                 " له لها لهم به بها بهم الیه الیهم ای علیهم علیکم منهم لکم قد".split())
VERBS = frozenset({"یکن", "یکون", "یقال", "یسأل", "یستدرک", "یقول", "تجعل"})
BEFORE_ARTICLE = frozenset("فی علی عن الی مع ایها ایتها".split())
# Persian words that end a run: vocatives, headings, credits.
BOUNDARIES = frozenset("ای همه مردمان به رسید گویمش مشهورترین شجاعترین تالیف تألیف نویسنده"
                       " نگارخانه منابع پانویس پانویسمنابع جستارهای وابسته تصحیحها اخترشناسی"
                       " موسیقی اعتقادات نگارش نویسندگی انتشارات چاپ جلد شماره ش ق مولانا"
                       " کافی اثر شرح".split())
# Short Arabic titles attested in development text, not a grammar of titles.
TITLES = ("روضة الصفا", "نفحات الانس", "کتاب التوحید", "مجمع الفصحاء", "اسرار التوحید",
          "آثار البلاد", "اخبار العباد", "عجائب المخلوقات", "غرائب الموجودات",
          "مفید العلوم", "مبید الهموم")
GAP = re.compile(r'[\n,.!?؟،؛;:«»"()\[\]{}/\\–—=]|[A-Za-z0-9۰-۹٠-٩]|(?<!\S)-(?!\S)|-(?=ت[أا]لیف)')


def quran_gap(text: str) -> bool:
    """Only horizontal space, direction marks, slash and dash may separate words."""
    return any(unicodedata.category(c) != 'Zs' and c not in '\t/—–-\u200e\u200f'
               for c in text)


@lru_cache(maxsize=32768)
def fold_word(word: str) -> str:
    """A lossy comparison key, never returned as text."""
    word = unicodedata.normalize("NFC", word)
    return MARKS.sub("", word).translate(FOLD).replace("ـ", "").replace("‌", "").replace("‍", "")


def _rows(name: str) -> list[str]:
    lines = (DATA / name).read_text(encoding="utf-8").splitlines()
    return [line.split("\t")[0] for line in lines if line and not line.startswith("#")]


def _veto(word: str) -> bool:
    return word in PERSIAN or word.startswith(("می‌", "نمی‌"))


@dataclass(frozen=True)
class ArabicSpan:
    start: int
    end: int
    reason: tuple[str, ...]
    quran_refs: tuple[str, ...] = ()
    kind: str = "passage"


class ArabicDetector:
    """Scores runs of words between Persian words and punctuation. A run longer
    than 18 words abstains; unsupported words at its edges are trimmed."""

    def __init__(self, *, persian=None, formulae=None):
        self.persian = frozenset(fold_word(w) for w in (_rows("arabic-persian.tsv") if persian is None else persian))
        self.formulae = frozenset(" ".join(fold_word(w) for w in TOKEN.findall(f))
                                  for f in (_rows("arabic-formulae.tsv") if formulae is None else formulae))
        self.titles = [tuple(fold_word(w) for w in t.split()) for t in TITLES]
        self.longest = max((len(f.split()) for f in self.formulae), default=2)
        self.phrase_starts = frozenset(f.split()[0] for f in self.formulae if f)

    def _features(self, token):
        return self._word_features(token[0])

    @lru_cache(maxsize=32768)
    def _word_features(self, raw):
        word = fold_word(raw)
        bare = word[1:] if word.startswith(("و", "ف")) and len(word) > 3 else word
        article = word.startswith("ال") and len(word) > 3 and word not in self.persian
        morph = word in VERBS or (len(word) >= 5 and word.startswith(("ی", "ت")) and word.endswith(("ون", "ین")))
        return (word in STRONG or bare in STRONG, word in WEAK or bare in WEAK, article, "ة" in word,
                len(MARKS.findall(raw)), morph)

    def _reasons(self, tokens, a, b):
        chunk = tokens[a:b]
        if len(chunk) < 2 or " ".join(fold_word(t[0]) for t in chunk) in self.formulae:
            return ()
        feats = [self._features(t) for t in chunk]
        strong, weak, article, teh, marks, morph = map(sum, zip(*feats))
        n = len(chunk)
        reasons = []
        if n >= 3 and strong >= 2:
            reasons.append("function_context")
        if n >= 3 and strong >= 1 and (article + teh + morph >= 1 or weak >= 2):
            reasons.append("function_plus_grammar")
        if n >= 3 and weak >= 2 and article + teh >= 2:
            reasons.append("weak_context_plus_nouns")
        if n >= 3 and weak >= 2 and teh and marks >= 2:
            reasons.append("marked_function_context")
        if n >= 3 and marks >= n and sum(f[4] > 0 for f in feats) >= 3 and (strong or article or (weak >= 2 and teh)):
            reasons.append("vocalized_syntax")
        if n >= 2 and article >= 1 and teh >= 1 and article + teh >= 3:
            reasons.append("arabic_nominal_construction")
        return tuple(reasons)

    def detect(self, text: str) -> list[ArabicSpan]:
        tokens = list(TOKEN.finditer(text))
        words = [fold_word(t[0]) for t in tokens]
        candidates = []
        starts = {title[0]: [] for title in self.titles}
        for i, word in enumerate(words):
            if word in starts:
                starts[word].append(i)
        for title in self.titles:
            for i in starts[title[0]]:
                end = i + len(title)
                if tuple(words[i:end]) == title and not any(
                        GAP.search(text[tokens[k].end():tokens[k + 1].start()]) for k in range(i, end - 1)):
                    candidates.append((i, end, ("attested_short_title",)))
        runs, start = [], 0
        for i, t in enumerate(tokens):
            veto = _veto(t[0]) or _veto(words[i]) or words[i] in BOUNDARIES or words[i].endswith(("ترین", "ترين"))
            if veto or (i > start and GAP.search(text[tokens[i - 1].end():t.start()])):
                if start < i:
                    runs.append((start, i))
                start = i + 1 if veto else i
        if start < len(tokens):
            runs.append((start, len(tokens)))
        for a, b in runs:
            if b - a > 18:
                continue
            while a < b and not any(self._features(tokens[a])):
                a += 1
            while b > a and not any(self._features(tokens[b - 1])):
                b -= 1
            if " ".join(words[a:b]) in self.formulae:
                continue
            if reasons := self._reasons(tokens, a, b):
                candidates.append((a, b, reasons))
        selected = []
        for a, b, reasons in sorted(candidates, key=lambda c: (c[0], -c[1])):
            start, end = tokens[a].start(), tokens[b - 1].end()
            # A span inside or overlapping one already taken is dropped, so a
            # title never pulls a scored boundary out.
            if any(s.start <= start and end <= s.end for s in selected) or (selected and start < selected[-1].end):
                continue
            selected.append(ArabicSpan(start, end, reasons))
        for a, b in self._phrases(text, tokens, words):
            start, end = tokens[a].start(), tokens[b - 1].end()
            if not any(s.start < end and start < s.end for s in selected):
                selected.append(ArabicSpan(start, end, ("set_phrase",), kind="phrase"))
        return sorted(selected, key=lambda s: s.start)

    def _phrases(self, text, tokens, words):
        found, i = [], 0
        while i < len(tokens) - 1:
            if words[i] not in BEFORE_ARTICLE and words[i] not in self.phrase_starts:
                i += 1
                continue
            for n in range(min(self.longest, len(tokens) - i), 1, -1):
                # علی is also the name, so a Persian ال word after it stays Persian (علی الهام).
                article = (n == 2 and words[i] in BEFORE_ARTICLE and words[i + 1].startswith("ال")
                           and len(words[i + 1]) > 3 and (words[i] != "علی" or words[i + 1] not in self.persian))
                if (article or " ".join(words[i:i + n]) in self.formulae) and not any(
                        GAP.search(text[tokens[k].end():tokens[k + 1].start()]) for k in range(i, i + n - 1)):
                    found.append((i, i + n))
                    i += n - 1
                    break
            i += 1
        return found


def quran_key(word: str) -> str:
    """The folded word, with the conjunction و dropped alone or split off."""
    key = fold_word(word)
    if key == "و":
        return ""
    return key[1:] if key.startswith("و") and len(key) >= 4 else key


class QuranIndex:
    """Maximal contiguous Quran wording, in copies given as ``sura|aya|text``
    lines (Tanzil's format). A match needs ``min_content`` words and lies within
    one verse; shorter verses abstain unless ``whole_short_verse``. A match
    wholly inside a formula is dropped. Exact means equality after folding
    harakat and letter variants, never dropping conjunctions. ``exact=False``
    restores the older conjunction-insensitive comparison."""

    def __init__(self, paths, *, min_content=3, whole_short_verse=False, formulae=None, exact=True):
        if min_content not in (3, 4):
            raise ValueError("min_content must be 3 or 4")
        self.min_content = min_content
        self.key = fold_word if exact else quran_key
        self.formulae = tuple(tuple(k for w in TOKEN.findall(f) if (k := self.key(w)))
                              for f in (_rows("arabic-formulae.tsv") if formulae is None else formulae))
        self.verses, self.index, self.short = [], {}, {}
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip().split("|", 2)
                    if len(parts) != 3 or not parts[0].isdigit():
                        continue
                    keys = tuple(k for t in TOKEN.findall(parts[2]) if (k := self.key(t)))
                    ref = f"{parts[0]}:{parts[1]}"
                    for j in range(len(keys) - min_content + 1):
                        self.index.setdefault(keys[j:j + min_content], []).append((len(self.verses), j))
                    if whole_short_verse and 2 <= len(keys) < min_content:
                        self.short.setdefault(keys, set()).add(ref)
                    self.verses.append((keys, ref))

    def detect(self, text: str) -> list[ArabicSpan]:
        tokens = list(TOKEN.finditer(text))
        keyed = [(i, k) for i, t in enumerate(tokens) if (k := self.key(t[0]))]
        keys = [k for _, k in keyed]
        candidates: dict[tuple[int, int], set[str]] = {}
        n = self.min_content
        for j, gram in enumerate(zip(*(keys[k:] for k in range(n)))):
            for vi, offset in self.index.get(gram, ()):
                verse, ref = self.verses[vi]
                end = j + n
                while (end < len(keys) and offset + end - j < len(verse)
                       and keys[end] == verse[offset + end - j]
                       and not quran_gap(text[tokens[keyed[end - 1][0]].end():tokens[keyed[end][0]].start()])):
                    end += 1
                candidates.setdefault((j, end), set()).add(ref)
        for size in (2, 3) if self.short else ():
            for j, gram in enumerate(zip(*(keys[k:] for k in range(size)))):
                if refs := self.short.get(gram):
                    candidates.setdefault((j, j + size), set()).update(refs)
        if not candidates:
            return []
        blocked = [(j, j + len(p)) for p in self.formulae if p
                   for j in range(len(keys) - len(p) + 1) if tuple(keys[j:j + len(p)]) == p]
        found = []
        for (j, end), refs in sorted(candidates.items(), key=lambda x: (x[0][0], -x[0][1])):
            if any(a <= j and end <= b for a, b in blocked):
                continue
            a, b = keyed[j][0], keyed[end - 1][0]
            # Words must not leap across a sentence, heading or gloss.
            if any(quran_gap(text[tokens[k].end():tokens[k + 1].start()]) for k in range(a, b)):
                continue
            start, stop = tokens[a].start(), tokens[b].end()
            if not any(s.start <= start and stop <= s.end for s in found):
                found.append(ArabicSpan(start, stop, ("quran_exact",), tuple(sorted(refs)), "quran"))
        # On an overlap the longer verified interval wins; none is extended.
        selected = []
        for span in sorted(found, key=lambda s: (-(s.end - s.start), s.start)):
            if not any(span.start < s.end and s.start < span.end for s in selected):
                selected.append(span)
        return sorted(selected, key=lambda s: s.start)


@lru_cache(maxsize=1)
def detector() -> ArabicDetector:
    return ArabicDetector()


def words(text: str, *, quran=None, threshold=None, strict=False):
    """Word decisions and calibrated probabilities, with original offsets.

    Tokens follow TOKEN (Arabic-script words, including internal ZWNJ); other
    scripts, digits and punctuation are excluded. A silver-trained model is
    calibrated on weighted, GPT-labeled development lines. Nonverified runs
    shorter than two words abstain, regardless of probability. Strict mode
    requires rule agreement and uses a higher default threshold.
    """
    from .arabic_model import model
    return model().words(text, quran=quran, threshold=threshold, strict=strict)


def spans(text: str, *, quran=None, strict=False, verse=False, threshold=None) -> list[ArabicSpan]:
    """Model spans; strict=True uses the normalizer threshold. verse is metadata."""
    from .arabic_model import model
    return model().detect(text, quran=quran, verse=verse, threshold=threshold, strict=strict)


def mask(text: str, *, quran=None, threshold=None, strict=False) -> list[bool]:
    """One boolean per TOKEN match, in original order (punctuation excluded)."""
    return [word.arabic for word in words(text, quran=quran, threshold=threshold, strict=strict)]
