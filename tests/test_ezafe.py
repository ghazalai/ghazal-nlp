import json

import pytest

from ghazal_nlp import ezafe
from ghazal_nlp.aruz import DATA


def segment(text):
    return {"source": "fixture", "segment_id": "s", "text": text}


def site(text, head, nxt):
    h0 = text.index(head)
    n0 = text.index(nxt, h0 + len(head)) if nxt else len(text)
    return {"neutralized_text": text, "neutral_head_span": [h0, h0 + len(head)], "neutral_next_span": [n0, n0 + len(nxt)]}


@pytest.mark.parametrize("text,mark", [
    ("خانهٔ من", "heh_hamza"), ("خانۀ من", "precomposed_heh"), ("خانه‌ی من", "zwnj_yeh"), ("خانهی من", "solid_yeh"),
    ("خانه ی من", "spaced_yeh"), ("خانه من", "unmarked"), ("خانه‌ي من", "zwnj_yeh"), ("خان\u06d5\u0654 من", "nfd_damaged_heh"),
    ("خانهِ من", "heh_kasra"),
])
def test_every_spelling_neutralises_to_one_junction(text, mark):
    row, = ezafe.sites(segment(text))
    assert (row["neutralized_text"], row["observed_mark_class"], row["original_junction"]) == ("خانه من", mark, text[:-2])
    assert ezafe.junction(row) == ezafe.Decision("ezafe", "pronoun")


def test_offsets_address_the_original_text():
    text = "😀 دِلِ خَانۀ مَن، روشن است."
    row, = ezafe.sites(segment(text))
    assert row["neutralized_text"] == "😀 دِلِ خَانه مَن، روشن است."
    assert text[slice(*row["head_span"])] == "خَانۀ" and text[slice(*row["next_span"])] == "مَن"
    assert row["neutralized_text"][slice(*row["neutral_head_span"])] == "خَانه"
    assert row["neutralized_text"][slice(*row["neutral_next_span"])] == "مَن"


def test_what_counts_as_a_site():
    row, = ezafe.sites(segment("خانه، من آمدم"))
    assert "،" in row["neutralized_text"] and not row["marked"]
    assert [r["head"] for r in ezafe.sites(segment("ماهی من، راهی دیگر، خانه‌ای بزرگ، رفته است، برده بود"))] == ["رفته", "برده"]
    first, second = ezafe.sites(segment("خانه من، خانه تو"))
    assert first["site_id"] != second["site_id"]
    assert next(ezafe.sites(segment("خانه من، خانه او")))["site_id"] != first["site_id"]


@pytest.mark.parametrize("text,head,nxt,reason", [
    ("و مانند خزانه که کنوز آنجا فرستادندی", "خزانه", "که", "connector"),
    ("خرقه را دام لقمه ساخته", "خرقه", "را", "connector"),
    ("تو زنده و پاینده", "زنده", "و", "connector"),
    ("بوسه بر کنار ساغر نه", "بوسه", "بر", "preposition"),
    ("که دارد سینه چون دیگ جوشان", "سینه", "چون", "simile"),
    ("چاره نیست بجز دیدن", "چاره", "نیست", "copula"),
    ("خوشکرده. خفته خیز", "خوشکرده", "خفته", "punctuation"),
    ("جامه", "جامه", "", "end"),
])
def test_a_written_mark_that_is_not_an_ezafe(text, head, nxt, reason):
    assert ezafe.written(site(text, head, nxt)) == ezafe.Decision("not_ezafe", reason)


@pytest.mark.parametrize("text,head,nxt", [
    ("ازمنه پیش از تاریخ", "ازمنه", "پیش"), ("مدیر روزنامه «ایران» بود", "روزنامه", "ایران"),
    ("کابینه دوم علم", "کابینه", "دوم"), ("شیوه برخورد با فرهنگ", "شیوه", "برخورد"),
])
def test_ordinary_written_ezafes_stay(text, head, nxt):
    assert ezafe.written(site(text, head, nxt)).label == "ezafe"


def test_the_verb_arm_flips_homographs():
    assert ezafe.written(site("شیوه برخورد با فرهنگ", "شیوه", "برخورد"), "closed+verb").label == "not_ezafe"


@pytest.mark.parametrize("text,label", [
    ("خانه را دیدم", "none"), ("خانه در شهر", "none"), ("خانه، من آمدم", "none"), ("رفته است", "none"),
    ("خانه می‌سوزد", "none"), ("گفته خودش", "ezafe"), ("خانه زیبا", "ezafe"),
])
def test_junctions(text, label):
    assert ezafe.junction(next(ezafe.sites(segment(text)))).label == label


def test_only_the_neutral_text_is_read():
    row = next(ezafe.sites(segment("خانه من")))
    assert ezafe.junction({**row, "marked": True, "head": "رفته", "text": "رفته است", "observed_mark_class": "heh_hamza"}) == ezafe.junction(row)
    end = {"neutralized_text": "خانه", "neutral_head_span": [0, 4], "neutral_next_span": [4, 4]}
    assert ezafe.junction(end).label == "none"
    with pytest.raises(ValueError):
        ezafe.junction({**end, "neutral_next_span": [3, 9]})


def test_the_development_rulings():
    written = [json.loads(line) for line in open(DATA / "ezafe-written-dev.jsonl", encoding="utf-8")]
    junctions = [json.loads(line) for line in open(DATA / "ezafe-junctions-dev.jsonl", encoding="utf-8")]
    assert ezafe.score(written, ezafe.written, "blind")["agree"] == 96
    assert ezafe.score(written, ezafe.written, "revised")["agree"] == 100
    assert ezafe.score(junctions, ezafe.junction, "blind")["agree"] == 111
    assert ezafe.score(junctions, ezafe.junction, "revised")["agree"] == 109


def test_junctions_in_plain_text():
    found = ezafe.junctions("نامه را نوشتم و خانهٔ زیبا و خانه ای دیدم")
    assert [(j.head, j.next, j.mark, j.label) for j in found] == [
        ("نامه", "را", "unmarked", "none"), ("خانه", "زیبا", "heh_hamza", "ezafe"), ("خانه", "ای", "unmarked", "none")]
    assert found[1].span == (16, 26)


def test_heads_and_neighbours_without_an_ezafe():
    labels = [j.label for j in ezafe.junctions("گرفته شده است. خیره شده بود. همیشه یک قدم جلوتر. تکه تکه کرد. از جمله آشنایی")]
    assert labels == ["none"] * 8


def test_the_verse_profile_reads_the_metre():
    metre = "فعولن فعولن فعولن فعل (متقارب مثمن محذوف یا وزن شاهنامه)"
    found = [(j.label, j.reason) for t in ("فرستاده شاه را پیش خواند", "از اندیشه دل دور کن تا توان", "خانه زیبا")
             for j in ezafe.junctions(t, "verse", metre)]
    assert found == [("ezafe", "metre"), ("none", "metre"), ("none", "default")]
    with pytest.raises(ValueError):
        ezafe.junction(next(ezafe.sites(segment("خانه زیبا"))), "poetry")
