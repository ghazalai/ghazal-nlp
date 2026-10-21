import pytest

from ghazal_nlp.arabic import STRONG, WEAK, ArabicDetector, QuranIndex, detector


def spans(text, d=None):
    return [text[s.start:s.end] for s in (d or detector()).detect(text)]


@pytest.mark.parametrize("text", [
    "من بعد پریشان نگویم", "من و ما و علی", "تیشة غمّاز راز کوهکن را فاش کرد", "علی از این خانه می‌رود",
    "البرز و الهام و التفات", "رَحْمَة", "ابن ادهم من به ترک ملک فانی گویمش",
])
def test_abstains_on_homograph_or_script_only(text):
    assert detector().detect(text) == []


@pytest.mark.parametrize("text", ["بسم الله الرحمن الرحیم", "علیه السلام", "ان شاء الله", "حق تعالی", "لا اله الا الله"])
def test_a_formula_is_a_phrase_not_a_passage(text):
    assert [(text[s.start:s.end], s.kind) for s in detector().detect(text)] == [(text, "phrase")]


@pytest.mark.parametrize("text, phrase", [
    ("نمی‌آید و مع ذلک هرگاه بندگان اشرف", "مع ذلک"), ("بخانه رو دله فی الحال مقنعهٔ سبز", "فی الحال"),
    ("صندل خادم گفت ایها الوزیر مگر او", "ایها الوزیر"), ("دیدم علی الهام را", None),
])
def test_set_phrases_and_a_preposition_before_the_article(text, phrase):
    assert [text[s.start:s.end] for s in detector().detect(text) if s.kind == "phrase"] == ([phrase] if phrase else [])


def test_persian_bounds_the_span_and_offsets_are_original():
    quote = "انّ امّتی ستفترق علی اثنتین و سبعین فرقة"
    (span,) = detector().detect(f"او گفت: «{quote}» و این سخن است.")
    assert f"او گفت: «{quote}» و این سخن است."[span.start:span.end] == quote and span.reason


def test_a_long_clause_with_an_honorific_is_not_a_formula():
    assert detector().detect("قال رسول الله صلی الله علیه و سلم انما الاعمال بالنیات")


def test_article_words_known_as_persian_are_injected():
    assert ArabicDetector(persian=()).detect("کلمة العربیة المدینة")
    assert ArabicDetector(persian=["العربیة", "المدینة"]).detect("کلمة العربیة المدینة") == []


def test_function_words_are_single_words():
    assert {"فانما", "فان", "لکن", "لعل"} <= STRONG and {"بین", "عند"} <= WEAK
    assert not {"فانمافان", "لکنلعل", "بینعند"} & (STRONG | WEAK)
    assert detector().detect("هذا هو الذی لم یکن له شریک")


@pytest.mark.parametrize("title", ["روضة الصفا", "نفحات الانس", "کتاب التوحید"])
def test_short_titles_are_bounded(title):
    assert spans(f"او در «{title}» چنین نوشته است.") == [title]


def test_vocatives_headings_and_credits_bound_a_span():
    text = "ای همه مردمان کُلُوا مِمَّا فِی الْأَرْضِ این سخن است."
    assert spans(text) and all("مردمان" not in s for s in spans(text))
    assert all("تالیف" not in s for s in spans("الشبیبی فی شبابه السیاسی -تالیف علی عبد شناوه"))
    text = "هذا هو الذی قال / علی عبد شناوه منابع"
    assert detector().detect(text) and all(s.end <= 16 for s in detector().detect(text))


def test_spans_do_not_overlap():
    text = "هذا هو الذی قال انما الاعمال بالنیات. این سخن است. لم یکن له شریک"
    found = detector().detect(text)
    assert len(found) == 2 and found[0].end <= found[1].start
    assert all(0 <= s.start < s.end <= len(text) for s in found)


def test_quran_matching(tmp_path):
    path = tmp_path / "quran.txt"
    original = "2|153|إن الله مع الصابرين\n74|21|ثم نظر\n2|156|إنا لله وإنا إليه راجعون\n# not a verse\n"
    path.write_text(original, encoding="utf-8")
    q = QuranIndex([path])
    text = "گفت: إنَّ الله مَعَ الصابرین و این سخن است."
    (span,) = q.detect(text)
    assert (text[span.start:span.end], span.quran_refs) == ("إنَّ الله مَعَ الصابرین", ("2:153",))
    assert path.read_text(encoding="utf-8") == original
    for text in ("انا لله و انا الیه راجعون", "إن الله، مع الصابرین", "إن الله\nمع الصابرین", "ثم نظر"):
        assert q.detect(text) == [], text
    assert q.detect("إن الله مع")
    assert not QuranIndex([path], min_content=4).detect("إن الله مع")
    assert QuranIndex([path], min_content=4, whole_short_verse=True).detect("ثم نظر")
    with pytest.raises(ValueError):
        QuranIndex([path], min_content=2)
