import re


FRONT = re.compile(r"\A---[ \t]*\r?\n.*?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S)
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
COMMENT = re.compile(r"<!--.*?-->|<(script|style|pre|textarea)\b.*?</\1[ \t]*>", re.S | re.I)
INDENTED = re.compile(r" {4}|\t")
WHOLE = re.compile(r" {0,3}(?:([-*_])(?:[ \t]*\1){2,}|=+|\[[^\]^\n][^\]\n]*\]:[ \t]*\S.*)[ \t]*")
DELIMITER = re.compile(r" {0,3}\|?[ \t]*+:?-+:?[ \t]*+(?:\|[ \t]*+:?-+:?[ \t]*+)*\|?[ \t]*+")
MARKS = re.compile(r"[ \t]*+(?:>[ \t]?)*[ \t]*(?:(#{1,6})|[-*+]|\d{1,9}[.)])(?=[ \t]|$)(?:[ \t]+\[[ xX]\](?=[ \t]))?[ \t]*|[ \t]*(?:>[ \t]?)*[ \t]*")
CLOSING = re.compile(r"(?<![ \t])[ \t]+(?:#+[ \t]*)?$")
TRAILING = re.compile(r"(?<![ \t])[ \t]+$")
PIPE = re.compile(r"[ \t]*(?<!\\)\|[ \t]*")
ATOMS = re.compile(r"(?P<x>(`+)(?!`).*?(?<!`)\2(?!`)|<(?:[a-z][a-z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>)|(?P<s></?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>)|\[\^[^\]\s]+\]:?", re.I)
HOLE = {"x": "x", "s": " "}  # code and URLs stand in the prose as a Latin word, HTML tags as a space
LINK = re.compile(r"(!?\[)[^\]\n]*(\]\((?:<[^>\n]*>|[^\s)]*)(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'))?[ \t]*\)|\]\[[^\]\n]*\])")
LOOSE = re.compile(r"(?P<x>(?:https?://|www\.)[^\s<>]*[^\s<>.,;:!?،؛؟«»\"')\]*_~]|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)|\\[!-/:-@\[-`{-~]|\*+|_+|~~")


def masks(text):
    taken, cut = bytearray(len(text)), []

    def drop(s, e, sep=""):
        if s < e and not any(taken[s:e]):
            taken[s:e] = b"\1" * (e - s)
            cut.append((s, e, sep))

    if m := FRONT.match(text):
        drop(0, m.end())
    lines, at = [], 0
    for line in text.split("\n"):
        lines.append((at, line))
        at += len(line) + 1
    fence, code, blank = None, False, True
    for s, line in lines:
        if fence or (m := FENCE.match(line)):
            drop(s, min(s + len(line) + 1, len(text)))
            if not fence:
                fence = m[1]
            elif line.strip().startswith(fence) and not line.strip().strip(fence[0]):
                fence = None
            code, blank = False, False
            continue
        code = bool(INDENTED.match(line)) and (code or blank) or code and not line.strip()
        if code:
            drop(s, min(s + len(line) + 1, len(text)))
        blank = not line.strip()
    for m in COMMENT.finditer(text):
        drop(*m.span())
    table = False
    for i, (s, line) in enumerate(lines):
        table = "|" in line and (table or i + 1 < len(lines) and "|" in lines[i + 1][1] and "-" in lines[i + 1][1] and DELIMITER.fullmatch(lines[i + 1][1]))
        if WHOLE.fullmatch(line) or table and "-" in line and DELIMITER.fullmatch(line):
            drop(s, min(s + len(line) + 1, len(text)))
            continue
        head = MARKS.match(line)
        drop(s, s + head.end())
        tail = CLOSING.search(line) if head[1] else TRAILING.search(line)
        if tail:
            drop(s + tail.start(), s + len(line))
        for m in PIPE.finditer(line) if table else ():
            drop(s + m.start(), s + m.end(), "\n")
        if all(taken[s:s + len(line)]):
            continue
        for m in ATOMS.finditer(line):
            drop(s + m.start(), s + m.end(), HOLE.get(m.lastgroup, ""))
        for m in LINK.finditer(line):
            drop(s + m.start(1), s + m.end(1))
            drop(s + m.start(2), s + m.end(2))
        for m in LOOSE.finditer(line):
            drop(s + m.start(), s + (m.start() + 1 if m[0][0] == "\\" else m.end()), HOLE.get(m.lastgroup, ""))
    return sorted(cut)


def view(text):
    """The prose of a Markdown text, and for each of its characters the place in the text (-1 where a cut stands)."""
    if "\r\n" in text:
        positions = [i for i, c in enumerate(text) if c != "\r" or text[i:i + 2] != "\r\n"]
        prose, at = view("".join(text[i] for i in positions))
        return prose, tuple(positions[i] if i >= 0 else -1 for i in at)
    out, at, i = [], [], 0
    for s, e, sep in masks(text):
        out += [text[i:s], sep]
        at += [*range(i, s), *[-1] * len(sep)]
        i = e
    return "".join(out) + text[i:], (*at, *range(i, len(text)))


def back(at, s, e):
    if e > s:
        return (at[s], at[e - 1] + 1) if at[s] >= 0 and at[e - 1] - at[s] == e - s - 1 else None
    p = at[s - 1] + 1 if s and at[s - 1] >= 0 else at[s] if s < len(at) and at[s] >= 0 else None
    return None if p is None else (p, p)
