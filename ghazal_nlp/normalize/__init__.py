import re
from functools import lru_cache

from . import fold, forms, repair

VERSION = "1.0.0"
KEY_VERSION = 2
HOUSE_VERSION = 2
EQUIVALENCE_VERSION = 2

STEPS = (fold.lines, fold.compose, fold.invisibles, fold.letters, fold.digits, fold.marks, fold.whitespace,
         repair.numbers, repair.teh_marbuta, repair.junctions, repair.mi, repair.punctuation)
VERSE = tuple(repair.verse_junctions if s is repair.junctions else s for s in STEPS)
BASIC = (fold.lines, fold.canonical, fold.controls, fold.whitespace)
PROFILES = {"fa": STEPS, "verse": VERSE, "arab": BASIC, "plain": BASIC, "none": ()}
ARAB = {"ar", "azb", "bqi", "ckb", "glk", "hac", "haz", "lki", "lrc", "mzn", "ps", "sdh", "tly", "ur"}
RAW = {"code", "example", "trajectory"}
FENCE = re.compile(r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", re.M | re.S)
INLINE = re.compile(r"`[^`\n]+`")
CODE = re.compile(r"^[ \t]+(?:(?:def|class) \w.*:|(?:if|elif|else|for|while|with|try|except)\b.*:|(?:return|import)\b.*"
                  r"|(.*(?:\w[ \t]*=[ \t]*\S.*|[A-Za-z_]\(.*|[{}])))[ \t]*$", re.M)
PERSIAN_WORD = re.compile(r"(?<!\S)[\u0621-\u06d3]")
OBJECT = "\ufffc"


# The profile for a row: Persian gets the steps above (a poem keeps ای apart), other Arabic-script languages and every
# other script only composition, control characters and whitespace, and code, examples and trajectories stay as written.
def profile(lang, kind):
    base, _, script = (lang or "fa").partition("-")
    if kind in RAW or base == "none":
        return "none"
    if base == "fa" and script in ("", "AF"):
        return "verse" if kind == "poem" else "fa"
    return "arab" if script == "Arab" or base in ARAB and script in ("", "Arab") else "plain"


# A step over the text outside code: ``` blocks, each run between them on its own, and `inline` spans, which the
# step sees as one placeholder; a text with code no fence marks is all code. Code stays as written but for its line
# endings, and so do the line breaks around a block.
def outside(step, text):
    if step is fold.lines:
        return step(text)
    if coded(text):
        return text
    if "`" not in text:
        return step(text)
    if "```" not in text:
        return inline(step, text)
    out, at = [], 0
    for m in [*FENCE.finditer(text), None]:
        part = text[at:m.start() if m else len(text)]
        core = part.strip("\n")
        new = inline(step, core) if core else ""
        lead, trail = part[:len(part) - len(part.lstrip("\n"))] if at else "", part[len(part.rstrip("\n")):] if m else ""
        out.append(lead + new + trail if new else "\n" if at and m else "")
        if m:
            out.append(m[0])
            at = m.end()
    return "".join(out)


# Two indented lines with a keyword or code syntax (a line of three Persian words or more is prose that mentions code).
@lru_cache(1)
def coded(text):
    if not ("\n " in text or "\n\t" in text or text[:1] in " \t"):
        return False
    lines = CODE.finditer(FENCE.sub("", text) if "```" in text else text)
    return sum(1 for m in lines if not m[1] or len(PERSIAN_WORD.findall(m[1])) < 3) >= 2


def inline(step, text):
    spans = INLINE.findall(text) if OBJECT not in text else []
    if not spans:
        return step(text)
    parts = step(INLINE.sub(OBJECT, text)).split(OBJECT)
    return "".join(p + c for p, c in zip(parts, spans + [""])) if len(parts) == len(spans) + 1 else step(text)


# The released corpus: the steps above, in order; a poem keeps ای after a word apart.
def normalize(text, verse=False):
    for step in VERSE if verse else STEPS:
        text = outside(step, text)
    return text


# The duplicate and search key: normalized text with every surface difference folded away.
def key(text):
    return fold.words(fold.collapse(fold.strip_marks(normalize(text))))


# Ghazal's own prose: normalized text with the house form wherever Persian has two correct forms; code stays as written.
def house(text):
    return outside(forms.choices, normalize(text))


# For scoring: two texts that differ only in accepted variants (هٔ or ه‌ی, کتابها or کتاب‌ها, ۱۲ or 12) are equal.
def equivalent(a, b):
    return outside(forms.variants, a) == outside(forms.variants, b)
