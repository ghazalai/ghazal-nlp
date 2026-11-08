import re
import unicodedata

ZWNJ, ZWJ = "\u200c", "\u200d"
LETTER = r"\u0621-\u064a\u066e-\u06d3\u06d5"
MARK = r"\u064b-\u0652\u0670"
HAMZA = r"\u0654\u0655"
# Letters that join the next one, and those that letters or teh_marbuta make joining (ي ك ى ە ۍ ة): a ZWNJ after
# them counts.
JOINER = "بپتثجچحخسشصضطظعغفقکگلمنهیئكيىڪھہەۍة"

PRESENTATION = re.compile("[\ufb50-\ufdcf\ufe70-\ufefc]+")
# A mark's presentation form is the mark alone, not the space or tatweel it decomposes to.
MARK_FORMS = {c: unicodedata.normalize("NFKC", chr(c)).lstrip(" ـ") for c in range(0xFE70, 0xFE80)}
SPACE = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
STAND_IN = "\u200e\u200f\u061c\u200b\u00ad\ufeff\u00ac\u009d"
MARK_IN_WORD = re.compile(f"[{STAND_IN}](?<=[{LETTER}{MARK}{HAMZA}][{STAND_IN}])[{STAND_IN}]*(?=[ـ{ZWNJ}{ZWJ}{STAND_IN}]*(?!ـ)[{LETTER}])")
INVISIBLE = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\u00ad\ufeff\u061c]")
JOINERS = f"[{ZWNJ}{ZWJ}]*{ZWJ}[{ZWNJ}{ZWJ}]*"
ARABIC_ZWJ = re.compile(f"(?<=[{LETTER}{MARK}{HAMZA}]){JOINERS}|{JOINERS}(?=[{LETTER}])")
ZWNJ_RUN = re.compile(f"{ZWNJ}+")
JOINED = re.compile(f"[{JOINER}ـ][{MARK}{HAMZA}]{{0,2}}$")
NEXT_LETTER = re.compile(f"[{LETTER}]")
LETTERS = {"ي": "ی", "ى": "ی", "ۍ": "ی", "ك": "ک", "ڪ": "ک", "ھ": "ه", "ہ": "ه", "ە": "ه"}
SEATED = r"\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed"
TATWEEL = re.compile(f"(?:(?<=[{JOINER}])|(?<=[{JOINER}][{MARK}{HAMZA}])|(?<=[{JOINER}][{MARK}{HAMZA}]{{2}}))ـ+(?![ـ{SEATED}])|ـ+(?![ـ{SEATED}])(?=[{LETTER}])")
# A line in another Arabic-script language: a letter of Kurdish, Urdu, Pashto, Sindhi or Uyghur that Persian lacks, or
# a Kurdish spelling (a word that starts with ئ, وو, یەک, ەکان, لە).
OTHER = re.compile("|".join([*"ڵڕۆێڤۊٹڈڑںےۓټډړږښګڼېڻڦڄڃڇڍڊڌڎڏۈۋ", r"ئ(?<![^\s«(\"']ئ)", "وو", "یەک", "ەکان", "ەکە", r"لە(?<!\Sلە)(?!\S)"]))
EZAFE_HAMZA = re.compile(f"(?<=ه)ء(?![{LETTER}{MARK}{HAMZA}])")
DOUBLED = re.compile(f"([{MARK}{HAMZA}])\\1+")
INNER = re.compile(r"(?<=\S)(?! (?![ \t\f\v]))[ \t\f\v]+")
SPACING = re.compile(r"[ \t\f\v](?:(?<=^.)|(?<=[\f\v])|[ \t\f\v])", re.M)
CODE = re.compile(r"[={}<>\[\]]|[A-Za-z_]\(")
LIST = re.compile(r"[-*•–+▪◦]|[0-9۰-۹]+[.)]")
ARABIC_LETTER = re.compile(f"[{LETTER}]")
BLANKS = re.compile(r"\n{3,}")

MARKS = re.compile(f"[{MARK}{HAMZA}]")
EZAFE = re.compile("\u06c0|\u0647(?:\u0654|\u200c\u06cc)")
COLLAPSE = str.maketrans({"ة": "ه", "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ی", "ء": "", ZWNJ: "",
                          **{p: a for p, a in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")}})
WORD = re.compile(r"[^\W_]+")


# CRLF and CR become LF.
def lines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


# NFC, with Arabic presentation forms and letter ligatures to plain letters (ﻛ → ک, ﻻ → لا) and wide or no-break
# spaces to a space. Other compatibility forms (…, ², ﷼, ﷺ) stay as written.
def compose(text):
    if PRESENTATION.search(text):
        text = PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m[0]), text.translate(MARK_FORMS))
    return unicodedata.normalize("NFC", SPACE.sub(" ", text))


# NFC: canonical composition only, for scripts whose compatibility forms are not ours to fold.
def canonical(text):
    return unicodedata.normalize("NFC", text)


def _zwnj(m):
    return ZWNJ if JOINED.search(m.string, max(0, m.start() - 4), m.start()) and NEXT_LETTER.match(m.string, m.end()) else ""


# Drops bidi controls, ZWSP, BOM and soft hyphen; ZWJ among the joiners next to Arabic script; any ZWNJ that changes
# no rendering; then a tatweel run joined to a letter, which only stretches the join (one standing alone or after a
# ZWNJ is a dash and stays: تهران ـ ایرنا, جناغی‌ـ ترقوه‌ای; one carrying a mark is its seat). A direction mark, ZWSP,
# soft hyphen, BOM or ¬ between two letters stood for a ZWNJ (حمل‌ونقل), so it becomes one. What comes together
# recomposes (و‌ٔ → ؤ).
def invisibles(text):
    text = INVISIBLE.sub("", MARK_IN_WORD.sub(ZWNJ, text))
    if ZWJ in text:
        text = ARABIC_ZWJ.sub(lambda m: m[0].replace(ZWJ, ""), text)
    text = ZWNJ_RUN.sub(_zwnj, text)
    if "ـ" in text:
        text = TATWEEL.sub("", text)
    return unicodedata.normalize("NFC", text)


# Drops bidi controls, ZWSP, BOM and soft hyphen only; joiners stay as written, for scripts whose joining we don't model.
def controls(text):
    return INVISIBLE.sub("", text)


# On lines not in another Arabic-script language, ي ى ۍ → ی, ك ڪ → ک, ھ ہ ە → ه and ء after ه → the ezāfe mark
# (خانهء → خانهٔ); ۀ and هٔ stay as written.
def letters(text):
    if not OTHER.search(text):
        return _letters(text)
    return "\n".join(line if OTHER.search(line) else _letters(line) for line in text.split("\n"))


def _letters(text):
    for a, b in LETTERS.items():
        text = text.replace(a, b)
    return EZAFE_HAMZA.sub("\u0654", text) if "ء" in text else text


# Arabic-Indic digits ٠-٩ → Persian ۰-۹; ASCII digits stay.
def digits(text):
    for a, b in zip("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹"):
        text = text.replace(a, b)
    return text


# Drops a mark written twice; strips none. A mark on nothing is spelled out or cited (غ َ م َ, تنوین (ٌ)) and stays.
def marks(text):
    return DOUBLED.sub(r"\1", text)


# On a Persian prose line (most of its letters Arabic script, no = { } < > [ ] and no name before a parenthesis) a
# run of spaces becomes one and the indentation goes unless a list marker follows (- * • – + ▪ ◦ 1. 1), glued to its
# text or not; tabs and other lines keep their spacing, which is code's or a table's. No spaces at line ends, at most
# one blank line, none at either end.
def whitespace(text):
    lines = [line.rstrip(" \t\f\v") for line in text.split("\n")]
    text = "\n".join(map(_spacing, lines) if SPACING.search(text) else lines)
    return BLANKS.sub("\n\n", text).strip("\n")


def _spacing(line):
    letters = SPACING.search(line) and not CODE.search(line) and sum(map(str.isalpha, line))
    if not letters or len(ARABIC_LETTER.findall(line)) * 2 <= letters:
        return line
    body = line.lstrip(" \t\f\v")
    lead = line[:len(line) - len(body)] if LIST.match(body) else ""
    return lead + INNER.sub(lambda m: m[0] if "\t" in m[0] else " ", body)


# Removes every harakat and hamza mark; the ه ezāfe in any spelling (ۀ, هٔ, ه‌ی) becomes bare ه.
def strip_marks(text):
    return MARKS.sub("", EZAFE.sub("ه", text))


# ة → ه, آ أ إ ٱ → ا, ؤ → و, ئ → ی, ء and ZWNJ dropped, Persian digits → ASCII.
def collapse(text):
    return text.translate(COLLAPSE)


# Casefolded words joined by single spaces; punctuation gone.
def words(text):
    return " ".join(WORD.findall(text.casefold()))
