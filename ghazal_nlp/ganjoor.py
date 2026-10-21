"""Poems from a Ganjoor API snapshot (``cat/<id>.json``), with the editors' marks.

Each section's ``htmlText`` keeps the harakat, ezāfe marks and ZWNJ that
``plainText`` drops; ``plainText`` gives the line structure. A poem's metre is
its sections' shared label: ``mixed`` when they disagree, ``absent`` when none
carries one. Its form and rhyme letters are the first section's.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

KEY = str.maketrans("يىكۀةؤئأإٱآ", "ییکههویاااا")
FORMS = {1: "ghazal", 2: "qasida", 3: "masnavi", 4: "qeta", 5: "robai", 6: "dobeyti", 10: "single", 16: "tarji"}


class _Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.depth += 1
            self.paragraphs.append("")
        elif tag == "br" and self.depth:
            self.paragraphs[-1] += "\n"

    def handle_endtag(self, tag):
        if tag == "p" and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.paragraphs[-1] += data


def _lines(text: str) -> list[str]:
    return [s for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines() if (s := line.strip())]


def html_lines(html: str) -> list[str]:
    parser = _Paragraphs()
    parser.feed(html)
    parser.close()
    return [line for p in parser.paragraphs for line in _lines(p)]


def _key(line: str) -> str:
    return "".join(c for c in line.translate(KEY) if unicodedata.category(c)[0] not in "CMPSZ")


def section_lines(plain: str, html: str) -> list[str]:
    """The html lines matched to the plain ones; a plain line html lacks is kept."""
    plain_lines, rich = _lines(plain), html_lines(html) if html.strip() else []
    if not rich or len(plain_lines) == len(rich):
        return rich or plain_lines
    out = []
    matcher = SequenceMatcher(a=[_key(x) for x in plain_lines], b=[_key(x) for x in rich], autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            out += rich[b0:b1]
        elif tag == "delete":
            out += [x for x in plain_lines[a0:a1] if not re.fullmatch(r"</?[A-Za-z][^<>]*>?", x)]
        elif tag == "replace" or any(_key(x) for x in rich[b0:b1]):
            raise ValueError(f"html and plain lines do not align: {len(rich)} rich, {len(plain_lines)} plain")
    return out


def poems(snapshot: Path):
    """Every poem once, in category order: id, poet, metre, metre_status, text,
    form, rhyme_letters."""
    seen = set()
    for path in sorted((snapshot / "cat").glob("*.json"), key=lambda p: int(p.stem)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        poet = (payload.get("poet") or {}).get("name") or ""
        for poem in (payload.get("cat") or {}).get("poems") or []:
            if poem.get("id") is None or poem["id"] in seen:
                continue
            seen.add(poem["id"])
            texts, metres = [], set()
            sections = poem.get("mainSections") or [{}]
            for section in sections:
                text = "\n".join(section_lines(str(section.get("plainText") or "").strip(), str(section.get("htmlText") or "")))
                if text:
                    texts.append(text)
                if metre := (section.get("ganjoorMetre") or {}).get("rhythm"):
                    metres.add(metre)
            status = "single" if len(metres) == 1 else "mixed" if metres else "absent"
            yield {"id": str(poem["id"]), "poet": poet, "metre": metres.pop() if status == "single" else None,
                   "metre_status": status, "text": "\n".join(texts),
                   "form": FORMS.get(sections[0].get("poemFormat")), "rhyme_letters": sections[0].get("rhymeLetters")}
