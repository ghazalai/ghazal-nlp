import re
from pathlib import Path

from . import verb
from .fold import HAMZA, JOINED, LETTER, MARK, MARKS, ZWNJ

DATA = Path(__file__).parents[1] / "data" / "normalize"

LATIN_MARK = re.compile(f"(?<=[{LETTER}{MARK}{HAMZA}\u200c])( *)([,;?]+)(?![A-Za-z])")
PERSIAN_MARK = str.maketrans(",;?", "،؛؟")
BEFORE_CLOSE = re.compile(f"(?: (?<=[\\w{MARK}{HAMZA}»)\\]] ) *(?=[،؛,.:!?؟»)\\]}}]*[،؛,.:!?؟»)\\]}}](?:\\s|$))| (?<=[{LETTER}{MARK}{HAMZA}] ) *"
                          f"(?=[،؛,.:!?؟»)\\]}}]*[،؛,.:!?؟][{LETTER}]))(?![:;=]-?[)(\\]\\[DPpOo3|/\\\\*])(?!\\. \\.|\\.{{4}})")
AFTER_OPEN = re.compile(f" (?<=[«(\\[{{] )(?<![\\w{MARK}{HAMZA}][«(\\[{{] ) *(?=[\\w{MARK}{HAMZA}])")
GLUE = re.compile(f"[.,،؛:!؟?»)\\]][{LETTER}]")
GLUED = re.compile(f"(?<!\\S)\\S*{GLUE.pattern}\\S*")
PUNCT = re.compile(f"(?<=[^\\s.,،؛:!؟?»)\\]])([.,،؛:!؟?»)\\]]+)(?=[{LETTER}])")
LATIN = re.compile("[A-Za-z@]|://|www\\.")
WORD_CHAR = re.compile(f"[{LETTER}{MARK}{HAMZA}\u200c]")
BOUND = re.compile("(?:ای|ام|اند|ایم|اید|ات|اش|ی|شان|تان|مان|ست|است|ها(?:ی(?:ی|ش|م|ت|مان|تان|شان)?|ست|شان|مان|تان)?)(?![\\w\u200c])")
TEH = dict(line.split("\t")[:2] for line in (DATA / "teh-marbuta.tsv").read_text().splitlines() if line and line[0] != "#")
TEH_WORD = re.compile(f"(?<![{LETTER}])[{LETTER}]*ة(?![{LETTER}{MARK}{HAMZA}])(?![ {ZWNJ}_]*ال)")
FUSED_WORDS = {line.split("\t")[0] for line in (DATA / "fused-words.tsv").read_text().splitlines() if line[:1] != "#"}
MI = re.compile(f"(نمی(?<![\\w{MARK}{HAMZA}\u200c\u200d]نمی)|می(?<![\\w{MARK}{HAMZA}\u200c\u200d]می)) ?(?=[^\\W\\d_])([آ-غف-ی][آ-غف-ی{MARK}{HAMZA}]+)(?!\\w)")
ENDINGS = {"ای", "ام", "اند", "ایم", "اید", "ات", "اش", "ی", "شان", "تان", "مان"}
NOT_A_STEM = {"که", "چه", "نه", "به", "ره", "گه", "مه", "شه", "ده", "زه"}
NOT_A_PLURAL_HOST = set("و که از به در با تا را یا هم نه بر برای چون اگر اما ولی پس نیز".split())
NOT_A_COMPOUND_HOST = set("به از در با بر تا که و یا را برای چون بی می نمی".split())
NOT_A_COMPOUND_TAIL = ENDINGS | {"ای", "گر", "ان", "که", "تر", "ترین", "ست", "یی"}
PLURAL = re.compile("ها(?:ی(?:ی|ش|م|ت|مان|تان|شان)?|ست|شان|مان|تان)?")
HEAD = f"(?<![\\w{MARK}{HAMZA}\u200c\u200d])"
WORD = f"[{LETTER}][{LETTER}{MARK}{HAMZA}\u200c]*+"
ENDING_SPACE = re.compile(f"{HEAD}({WORD})(?<=ه) (?=((?:{'|'.join(ENDINGS)})\\S*))")
VERSE_ENDING_SPACE = re.compile(f"{HEAD}({WORD})(?<=ه) (?=((?:{'|'.join(ENDINGS - {'ای'})})\\S*))")
DIGIT = "0-9۰-۹"
TOKEN_CHAR = f"{LETTER}{MARK}{HAMZA}{DIGIT}\u200c"
NUMBER = re.compile(f"(?<![A-Za-z0-9_#@/\\-+{TOKEN_CHAR}])(?:(?<![{DIGIT}][.:])|(?=[{LETTER}]))[{TOKEN_CHAR}]*[{DIGIT}][{TOKEN_CHAR}]*(?![A-Za-z0-9_#@/:\\-+{TOKEN_CHAR}])")
RUN = re.compile(f"[{DIGIT}]+|[^{DIGIT}]+")
DIGIT_CHARS = set("0123456789۰۱۲۳۴۵۶۷۸۹")
GLUED_DIGIT = re.compile(f"[{DIGIT}](?:(?<=[{LETTER}{MARK}{HAMZA}\u200c][{DIGIT}])|(?=[{LETTER}{MARK}{HAMZA}\u200c]))")
NUMBER_BOUND = set("""ساله سالگی روزه ماهه ساعته دقیقه‌ای ثانیه‌ای تایی گانه نفره طبقه لایه برابر درصدی رقمی ستاره بعدی جلدی
کیلویی متری گرمی لیتری تومانی ریالی دلاری ضلعی امتیازی ام امین امی م مین ها های هایی""".split())
NUMBER_TAIL = re.compile(f"(?:{'|'.join(sorted(NUMBER_BOUND, key=len, reverse=True))})(?:ی|ش|م|ت|مان|تان|شان|\u200c?ها|\u200c?های|\u200c?ای)?(?:\u200c.*)?")
PLURAL_SPACE = re.compile(f"{HEAD}({WORD}) (?=(ها\\S*))")
FOLLOWER = re.compile(f"[\\w{MARK}{HAMZA}\u200c]+")
NEXT_WORD = re.compile(f"\\s+([\\w{MARK}{HAMZA}\u200c]+)")
PAIR_WORD = f"(?:[^\\W\\d_]|\u0654)+(?:\u200c(?:[^\\W\\d_]|\u0654)+)*"
ARABIC_WORD = re.compile(f"[{LETTER}\u0654\u200c]*[{LETTER}][\u0654\u200c]*[{LETTER}][{LETTER}\u0654\u200c]*")
FLUSH = re.compile(f"(?<![{LETTER}\u200c])([{LETTER}]{{2,}}(?<!ا)ه)(ها[{LETTER}]*)")
COMPOUNDS = {tuple(line.split("\t")[:2]) for line in (DATA / "compounds.tsv").read_text().splitlines() if line[:1] != "#"}


# One regex that matches exactly the given words, as a trie, so it stays fast with thousands of them.
def trie(words):
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = {}

    def rx(node):
        alts = [re.escape(ch) + rx(sub) for ch, sub in node.items() if ch]
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return ("(?:" + body + ")?" if "" in node else body) if alts else ""
    return rx(root)


PAIR = re.compile(f"(?<![\\w{MARK}\u200c])(?={trie({f'{a} {b}' for a, b in COMPOUNDS})})([^ ]+) (?=({PAIR_WORD})(?![\\w{MARK}\u200c]))")


# ZWNJ after a letter that joins leftwards, nothing after one that doesn't (a ZWNJ there is invisible noise).
def join(host):
    return host + ZWNJ if JOINED.search(host) else host


def left(text, end):
    start = end
    while start and WORD_CHAR.match(text, start - 1):
        start -= 1
    return text[start:end]


def _space_after(m):
    run, token = m[1], m.string
    if run[-1] in "»)]" and BOUND.match(token, m.end()):
        return run
    if run == "." and sum(map(str.isalpha, left(token, m.start()))) == 1:
        return run
    return run + " "


def _glued(m):
    return m[0] if LATIN.search(m[0]) else PUNCT.sub(_space_after, m[0])


# A Latin comma, semicolon or question mark, or a run of them, after a Persian word → ، ؛ ؟, unless a Latin letter
# follows; then no space between a word and the ، ؛ , . : ! ? ؟ » ) ] } that end it, or between « ( [ { and the word
# they open, except in an emoticon, a path, a ratio, a spaced run of dots or a blank of four or more (:) ./ ۵ :۳ . . .
# ....), where the marks face the other way (»ایران« است) and before a mark glued to the next word unless both words
# are Persian; a space after punctuation between a word and a letter glued to it, except in URLs and Latin tokens,
# after an initial (م.حسینی) and before an ending bound to a closer («کتاب»ها).
def punctuation(text):
    if "," in text or ";" in text or "?" in text:
        text = LATIN_MARK.sub(lambda m: m[1] + m[2].translate(PERSIAN_MARK), text)
    text = AFTER_OPEN.sub("", BEFORE_CLOSE.sub("", text))
    return GLUED.sub(_glued, text) if GLUE.search(text) else text


def _number(m):
    parts = RUN.findall(m[0])
    out = parts[0]
    for a, b in zip(parts, parts[1:]):
        number = a[0] in DIGIT_CHARS
        word = b if number else a
        if (len(word) >= 2 or word == "و") and ZWNJ not in (word[0], word[-1]) and not (number and NUMBER_TAIL.fullmatch(b)):
            out += " "
        out += b
    return out


# A space between a number and a Persian word glued to it (۲۰هزار → ۲۰ هزار, سال۱۴۰۰ → سال ۱۴۰۰), except a compound
# element the number makes one word with, its endings too (۴۷ساله, ۵ام, ۸۰ها, ۳تاییش), a single letter other than و (۱۳۴۸ش), and tokens with
# Latin letters or _ # @ / : - + (codes, hashtags, dates and times).
def numbers(text):
    return NUMBER.sub(_number, text) if GLUED_DIGIT.search(text) else text


# ة → ه or ت in the 1,580 Persian words the table names, where the word ends there; every other ة stays, Arabic
# included: a tanwin on it (عادةً), an article after it (رحمة الله, مقبرةالشهدا).
def teh_marbuta(text):
    return TEH_WORD.sub(lambda m: TEH.get(m[0], m[0]), text) if "ة" in text else text


# می / نمی, spaced or fused, + verb → joined with ZWNJ when the verb analyzer accepts the verb, except a fused form
# that is a word of its own (fused-words.tsv: میدانی, میشی).
def mi(text):
    return MI.sub(lambda m: m[1] + ZWNJ + m[2] if verb.imperfective(MARKS.sub("", m[2])) and MARKS.sub("", m[0]) not in FUSED_WORDS
                  else m[0], text)


def _ending(m):
    host, follower = m[1], FOLLOWER.match(m[2])[0]
    if host.endswith(ZWNJ) or host in NOT_A_STEM or host.endswith("اه") or follower not in ENDINGS:
        return m[0]
    return join(host)


def _plural(m):
    host, follower = m[1], FOLLOWER.match(m[2])[0]
    after = NEXT_WORD.match(m.string, m.end() + len(m[2]))
    if (sum(map(str.isalpha, host)) < 2 or host.endswith(ZWNJ) or host in NOT_A_PLURAL_HOST
            or not PLURAL.fullmatch(follower) or follower in (host, after and after[1])):
        return m[0]
    return join(host)


def _pair(m):
    a, b = m[1], m[2].partition(ZWNJ)[0]
    if ((a, b) not in COMPOUNDS or a in NOT_A_COMPOUND_HOST or b in NOT_A_COMPOUND_TAIL or PLURAL.fullmatch(b)
            or not (ARABIC_WORD.fullmatch(a) and ARABIC_WORD.fullmatch(m[2]))):
        return m[0]
    return join(a)


# Word junctions, until nothing changes: the plural written flush against ه (خانهها), then a space → ZWNJ before
# an ending after ه (خانه ای), a plural (کتاب ها) and an attested compound (بین المللی).
def junctions(text, endings=ENDING_SPACE):
    while True:
        out = endings.sub(_ending, FLUSH.sub(f"\\1{ZWNJ}\\2", text) if "هها" in text else text)
        out = PAIR.sub(_pair, PLURAL_SPACE.sub(_plural, out) if " ها" in out else out)
        if out == text:
            return out
        text = out


# The junctions in verse, where ای after a word is mostly the vocative (پندم مده ای دوست): it stays apart.
def verse_junctions(text):
    return junctions(text, VERSE_ENDING_SPACE)
