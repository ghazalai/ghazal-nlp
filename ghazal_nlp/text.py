"""The input fold the analysers share: encodings a keyboard, a PDF or an old
code page produce, folded to one spelling. Arabic presentation forms become
letters, ي ى ك become ی ک, a no-break space a space, and a ZWJ goes."""

import re
import unicodedata

PRESENTATION = re.compile("[ﭐ-﷏ﹰ-ﻼ]+")
# A mark's presentation form is the mark alone, not the space or tatweel NFKC gives.
MARK_FORMS = {c: unicodedata.normalize("NFKC", chr(c)).lstrip(" ـ") for c in range(0xFE70, 0xFE80)}
LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "\xa0": " ", "‍": None})


def fold(text: str) -> str:
    text = PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m[0]), text.translate(MARK_FORMS))
    return unicodedata.normalize("NFC", text.translate(LETTERS))
