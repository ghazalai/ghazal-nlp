import re
from pathlib import Path

from .fold import LETTER, MARK, ZWNJ, compose, invisibles, lines
from .repair import trie

DATA = Path(__file__).parents[1] / "data" / "normalize"
EDGE = f"[\\w{MARK}{ZWNJ}]"
SPELL = {"spaced": " ", "zwnj": ZWNJ, "solid": ""}
FORMS = {}
for line in (DATA / "house.tsv").read_text().splitlines():
    if line[:1] != "#":
        a, b, form, others = line.split("\t")[:4]
        for other in others.split(","):
            FORMS.setdefault(a + SPELL[other] + b, a + SPELL[form] + b)
PAIR = re.compile(f"(?<!{EDGE})({trie(FORMS)})(?!{EDGE})")
EZAFE = re.compile(f"(?:ۀ|هٔ)(?![{LETTER}{MARK}])")
COPULA = re.compile(f"{ZWNJ}(است)(?!{EDGE})")
SUFFIX = re.compile(f"(?<=[{LETTER}])(?<!ه){ZWNJ}(?=(?:ها|های|هایی|تر|ترین|تری)(?!{EDGE}))")
LATIN = re.compile(r"[A-Za-z@#_=&]|://")
LATIN_LETTER = re.compile("[A-Za-z]")
ARABIC = re.compile(f"[{LETTER}]")
DIGIT = re.compile("[0-9]")
DIGIT_RUN = re.compile(r"\S*[0-9]\S*")
QUOTED = re.compile(f"[\"“]([^\"“”\\n]*[{LETTER}][^\"“”\\n]*)[\"”]")
DOTS = re.compile(r"\.(?: ?\.){2,}")
TO_PERSIAN = str.maketrans("0123456789%", "۰۱۲۳۴۵۶۷۸۹٪")
HARAKAT = re.compile("[َ-ْٰ]")
PLAIN = str.maketrans({**dict(zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")), "٪": "%", "«": '"', "»": '"',
                       "“": '"', "”": '"', "…": "...", "–": "-", "—": "-", "−": "-", "٫": "/"})
EZAFES = re.compile(f"ۀ|ه(?:ٔ|{ZWNJ}ی)(?![{LETTER}{MARK}])")


def _digits(m):
    return m[0] if LATIN.search(m[0]) else m[0].translate(TO_PERSIAN)


def _line(line):
    return DIGIT_RUN.sub(_digits, line) if len(ARABIC.findall(line)) > len(LATIN_LETTER.findall(line)) else line


# The house form wherever Persian has two correct forms: the ezāfe after ه as ه‌ی, است apart from its word, the
# pairs careful writers write more than one way as house.tsv writes them, Persian digits and ٪ on Persian lines
# outside Latin tokens, « » around a Persian quotation, … for three dots.
def choices(text):
    text = PAIR.sub(lambda m: FORMS[m[1]], COPULA.sub(r" \1", EZAFE.sub(f"ه{ZWNJ}ی", text)))
    text = "\n".join(map(_line, text.split("\n"))) if DIGIT.search(text) else text
    return DOTS.sub("…", QUOTED.sub(r"«\1»", text))


# One spelling for every accepted variant, for scoring: the ezāfe in any spelling as bare ه, harakat other than tanwin
# dropped, the house pairs, ZWNJ before a plural or comparative, any digits as ASCII, one form of each quote, dash,
# ellipsis and percent sign. A form Ghazal repairs (ي, a spaced می) stays distinct.
def variants(text):
    text = invisibles(compose(lines(text)))
    text = PAIR.sub(lambda m: FORMS[m[1]], COPULA.sub(r" \1", EZAFES.sub("ه", HARAKAT.sub("", text))))
    return re.sub(r"[ \t]+", " ", SUFFIX.sub("", text).translate(PLAIN)).strip()
