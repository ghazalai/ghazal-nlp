from functools import cache
from pathlib import Path

DATA = Path(__file__).parents[1] / "data" / "normalize"
PRESENT = ("م", "ی", "د", "یم", "ید", "ند")
PAST = ("", "م", "ی", "یم", "ید", "ند")
PAST_STEMS, PRESENT_STEMS, GLIDES = {}, {}, set()
COLLOQUIAL = {line.split("\t")[0] for line in (DATA / "colloquial-verbs.tsv").read_text().splitlines() if line[:1] != "#"}

for line in (DATA / "verb-stems.tsv").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        infinitive, past, present = line.split("\t")[:3]
        for s in past.split("|"):
            PAST_STEMS.setdefault(s, set()).add(infinitive)
        for s in present.split("|"):
            PRESENT_STEMS.setdefault(s, set()).add(infinitive)
            if s + "ی" in present.split("|"):
                GLIDES.add((infinitive, s))


# True when the word after می/نمی is a verb form: past stem + ending, past stem + ه, present stem + ending
# (a stem with a glide alternate must use it before ی), bare توان, or a colloquial form (تونی، شه).
@cache
def imperfective(body):
    if body == "توان" or body in COLLOQUIAL or body.endswith("ه") and body[:-1] in PAST_STEMS:
        return True
    if any(body.endswith(e) and len(body) > len(e) and body[: len(body) - len(e)] in PAST_STEMS for e in PAST):
        return True
    return any(body.endswith(e) and (stem := body[: -len(e)]) and any(
        e[0] != "ی" or (i, stem) not in GLIDES for i in PRESENT_STEMS.get(stem, ())) for e in PRESENT)
