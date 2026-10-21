import unicodedata
from collections import Counter

import pytest

from ghazal_nlp import vowels
from ghazal_nlp.vowels import (
    DAMMA, DAMMATAN, FATHA, FATHATAN, HAMZA, KASRA, SHADDA, SUKUN, TATWEEL, ZWNJ, Lexicon, build_lexicon,
    canonicalize, measure, readings_of, refusal_census, review_queue, skeleton, split_terminal,
)

KARM, KARAM = "ک" + FATHA + "ر" + SUKUN + "م", "ک" + FATHA + "ر" + FATHA + "م"


def readings(*forms):
    return tuple(r.form for r in readings_of(Counter(forms)))


@pytest.fixture
def small():
    return build_lexicon(Counter({
        "گ" + KASRA + "ل": 6, "گ" + DAMMA + "ل": 4, "گل": 100, "خ" + FATHA + "انه": 3, "خانه": 50,
        "سر" + KASRA: 9, "سر": 40, "درخت": 20,
    }))


def test_nfd_would_turn_persian_heh_into_kurdish_ae():
    assert unicodedata.normalize("NFD", "ۀ") == "ە" + HAMZA
    assert unicodedata.normalize("NFC", "ە" + HAMZA) != "ه" + HAMZA
    assert unicodedata.normalize("NFC", "ه" + HAMZA) == "ه" + HAMZA
    assert canonicalize("خانۀ") == canonicalize("خانه" + HAMZA) == "خانه" + HAMZA
    assert canonicalize("دیدە" + HAMZA) == "دیده" + HAMZA


def test_canonicalisation():
    assert canonicalize("ش" + DAMMA + DAMMA + "کر") == "ش" + DAMMA + "کر"
    assert canonicalize("شک" + SHADDA + FATHA + "ر") == canonicalize("شک" + FATHA + SHADDA + "ر") == "شک" + FATHA + SHADDA + "ر"
    assert canonicalize("فِي") == "فِی" and canonicalize("كتاب") == "کتاب"
    assert canonicalize("کـتاب") == "کتاب" and canonicalize(TATWEEL) == TATWEEL
    assert canonicalize("دل" + ZWNJ + KASRA) == "دل" + KASRA
    assert canonicalize("ام" + ZWNJ + DAMMA + SHADDA + "الخبائث") == "ام" + DAMMA + SHADDA + ZWNJ + "الخبائث"
    assert canonicalize(KASRA + "سر") == "سر" and canonicalize("کتاب" + ZWNJ) == "کتاب"
    assert canonicalize("مسا" + HAMZA + "له") == "مسأله"
    assert canonicalize(KASRA) == KASRA and canonicalize(ZWNJ) == ZWNJ
    assert canonicalize("شا" + ZWNJ + HAMZA + "ن" + KASRA) == "شأ" + ZWNJ + "ن" + KASRA


@pytest.mark.parametrize("token", [
    "شُُکر", "شک" + SHADDA + FATHA + "ر", "دل" + ZWNJ + KASRA, "خانۀ", "فِي", "کـتاب", KASRA + "سر",
    "ام" + ZWNJ + DAMMA + SHADDA + "الخبائث", TATWEEL, "شا" + ZWNJ + HAMZA + "ن" + KASRA,
])
def test_canonicalise_is_idempotent(token):
    assert canonicalize(canonicalize(token)) == canonicalize(token)


def test_what_canonicalisation_refuses():
    assert SHADDA in canonicalize("ش" + SHADDA + "کر")
    assert len(readings(KARM, KARAM)) == 2 and len(readings(KARM.replace(SUKUN, ""), KARAM)) == 1
    for left, right in [("رحمة", "رحمه"), ("مسأله", "مساله"), ("آرد", "ارد"), ("می" + ZWNJ + "رود", "میرود")]:
        assert skeleton(canonicalize(left)) != skeleton(canonicalize(right))


def test_only_different_vowels_on_one_letter_make_two_readings():
    shukr = "ش" + DAMMA + "ک" + SUKUN + "ر"
    assert readings("ش" + DAMMA + "کر", shukr) == (shukr,)
    assert len(readings("شک" + SHADDA + "ر", "شک" + FATHA + "ر")) == 1
    assert len(readings("ش" + DAMMA + "کر", "ش" + KASRA + "کر")) == 2
    assert len(readings("ش" + FATHA + "ک" + FATHA + SHADDA + "ر", "ش" + KASRA + "ک" + FATHA + SHADDA + "ر")) == 2
    assert len(readings("و" + FATHA, "و" + DAMMA)) == 2


def test_a_partial_form_joins_the_reading_that_carries_its_marks():
    sugar = "ش" + KASRA + "ک" + FATHA + SHADDA + "ر"
    result = readings_of(Counter({"شک" + SHADDA + "ر": 373, "ش" + DAMMA + "ک" + SUKUN + "ر": 5, sugar: 26}))
    assert next(r for r in result if r.form == sugar).count == 399
    forms = {"ش" + DAMMA + "کر": 3, "شک" + SUKUN + "ر": 1}
    assert all(r.form in forms for r in readings_of(Counter(forms)))


def test_final_letter_marks_are_syntax():
    assert split_terminal(canonicalize("سر" + KASRA)) == ("سر", KASRA)
    assert split_terminal(canonicalize("خانۀ")) == ("خانه", HAMZA)
    assert split_terminal(canonicalize("حق" + SHADDA + KASRA)) == ("حق" + SHADDA, KASRA)
    assert split_terminal("و" + FATHA) == ("و" + FATHA, "")
    assert split_terminal(canonicalize("اصلا" + FATHATAN)) == ("اصلا" + FATHATAN, "")
    assert split_terminal(canonicalize("س" + KASRA + "ر" + SHADDA + DAMMATAN)) == ("س" + KASRA + "ر" + SHADDA, DAMMATAN)
    assert [vowels.TERMINAL[m] for m in KASRA + HAMZA + DAMMATAN + SUKUN] == ["ezafe", "ezafe_he", "i3rab", "pausal"]


def test_the_lexicon(small):
    assert "درخت" not in small and small.lookup("درخت") is None
    assert "سر" not in small
    assert small["خانه"].settled and not small["گل"].settled
    assert [e.skeleton for e in small.ambiguous()] == ["گل"] and small["گل"].total == 110
    assert small["گل"].ambiguous(min_evidence=4) and not small["گل"].ambiguous(min_evidence=5)
    assert small.lookup("گل" + KASRA) is small.lookup("گ" + DAMMA + "ل") is small["گل"]


def test_measure_and_queue(small):
    m = measure(Counter({"گل": 10, "خانه": 5, "درخت": 5}), small)
    assert (m["tokens"], m["ambiguous"], m["settled"], m["unattested"], m["ambiguous_share"]) == (20, 10, 5, 5, 0.5)
    assert [(k, reason) for k, reason, _, _ in review_queue(small)] == [("گل", "rare_reading")]
    counts = Counter({KARM: 3, KARAM: 3})
    census = refusal_census(counts, build_lexicon(counts))
    assert census["sukun_deletion_would_merge_skeletons"] == 1
    assert set(census) >= {f"{name}_would_merge_skeletons" for name in vowels.REFUSED_FOLDS}


def test_the_lexicon_round_trips(small, tmp_path):
    vowels.write(small, review_queue(small), {}, tmp_path)
    assert Lexicon.load(tmp_path / "vowels-lexicon.tsv") == small


def test_the_shipped_lexicon():
    lex = vowels.lexicon()
    assert not lex["گل"].settled and not lex["شکر"].settled and lex["گل"].ezafe > 0
    assert len(lex.settled()) / len(lex) > 0.85
    assert "# source: Ganjoor" in (vowels.DATA / "vowels-lexicon.tsv").read_text(encoding="utf-8")
