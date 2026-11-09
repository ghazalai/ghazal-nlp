import gzip
import json
from pathlib import Path

import pytest

from ghazal_nlp import arabic
from ghazal_nlp.arabic import STRONG, WEAK, ArabicDetector, QuranIndex, detector
from ghazal_nlp.arabic_eval import annotated
from ghazal_nlp.arabic_model import WordModel


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


def test_word_model_context_and_offsets():
    from ghazal_nlp import arabic

    text = '🙂 علم و ثلث را به کار برد و مع‌الوصف درست نبود.'
    words = arabic.words(text)
    assert len(words) == len(arabic.TOKEN.findall(text))
    assert not any(arabic.mask(text))
    assert all(0 <= w.probability <= 1 for w in words)
    assert [text[w.start:w.end] for w in words] == arabic.TOKEN.findall(text)
    quote = 'إن العلم نور والجهل ظلام'
    text = f'او گفت: «{quote}» و رفت.'
    assert [text[s.start:s.end] for s in arabic.spans(text, threshold=.1)] == [quote]
    assert not arabic.words('رَحْمَة')[0].arabic
    assert arabic.words('')[0:] == []


def test_model_strict_mode_and_verse_metadata():
    from ghazal_nlp import arabic

    text = 'هذا هو الذی لم یکن له شریک'
    strict = arabic.spans(text, strict=True)
    rules = arabic.detector().detect(text)
    assert strict and all(any(r.start <= s.start < s.end <= r.end for r in rules) for s in strict)
    assert {s.kind for s in arabic.spans(text, verse=True)} == {'verse'}
    assert arabic.spans('بسم الله الرحمن الرحیم')[0].kind == 'formula'


def test_exact_quran_preserves_conjunctions_and_boundaries(tmp_path):
    from ghazal_nlp import arabic

    path = tmp_path / 'quran.txt'
    path.write_text('2|153|إن الله مع الصابرين\n', encoding='utf-8')
    q = QuranIndex([path], formulae=())
    assert not q.detect('إن الله و مع الصابرین')
    assert QuranIndex([path], formulae=(), exact=False).detect('إن الله و مع الصابرین')
    text = 'او گفت: إن الله مع الصابرین و رفت.'
    words = arabic.words(text, quran=q)
    assert [text[w.start:w.end] for w in words if w.kind == 'quran'] == ['إن', 'الله', 'مع', 'الصابرین']
    exact = [s for s in arabic.spans(text, quran=q) if s.kind == 'quran']
    assert len(exact) == 1 and text[exact[0].start:exact[0].end] == 'إن الله مع الصابرین'
    assert q.detect('إن الله مع، الصابرین')[0].end == len('إن الله مع')


def test_complete_annotation_evaluation():
    from ghazal_nlp.arabic import ArabicSpan
    from ghazal_nlp.arabic_eval import annotated

    rows = [{'text': 'الف ب ج', 'spans': [[0, 5, 'passage']], 'category': 'wiki'}]
    result = annotated(rows, lambda text: [ArabicSpan(4, 7, ())])
    counts = result['categories']['all']
    assert (counts['tp'], counts['fp'], counts['fn']) == (1, 1, 1)
    assert counts['span_precision'] == counts['span_recall'] == 0


@pytest.fixture
def index(tmp_path):
    path = tmp_path / 'quran.txt'
    path.write_text('1|1|هذا باب جميل اليوم\n', encoding='utf-8')
    return arabic.QuranIndex([path], formulae=())


@pytest.mark.parametrize('text', ['هذا باب جميل هذا باب جميل', 'هذا / باب جميل', 'هذا — باب جميل'])
def test_a1_verified_intervals_are_emitted_unchanged(index, text):
    expected = index.detect(text)
    for strict in (False, True):
        actual = [s for s in arabic.spans(text, quran=index, strict=strict) if s.kind == 'quran']
        assert actual == expected
    words = arabic.words(text, quran=index)
    assert {w.verified for w in words if w.verified} == {(s.start, s.end) for s in expected}


def test_model_has_per_source_provenance():
    with gzip.open(arabic.DATA / 'arabic-model.json.gz', 'rt') as f:
        data = json.load(f)
    provenance = data['provenance']
    assert set(provenance['sources']) <= set(provenance['source_details'])
    for source in provenance['source_details'].values():
        assert source['version'] and source['attribution'] and source['license']
    assert all(not any(c.isspace() for c in w) for w in data['words'])
    assert 'arabic-model.json.gz' in (arabic.DATA / 'LICENSE').read_text()
    assert {'naskban-ocr', 'tasnim', 'telegram-channels'} <= set(provenance['source_roles']['calibration'])
    assert 'tashkeela' not in provenance['sources']
    assert provenance['source_roles']['exact_silver_labels'] == ['tanzil']
    for name in ('arabic-formulae.tsv', 'arabic-persian.tsv'):
        assert '# Source:' in (arabic.DATA / name).read_text()


@pytest.mark.parametrize('inserted', ['слово', 'λέξη', '中文', '123', '۱۲۳', '🙂', '\n', '\r', '\x85', '\x1f', '\u2028', '\u200c'])
def test_a3_exact_match_cannot_skip_inserted_content(index, inserted):
    text = f'هذا {inserted} باب جميل'
    assert index.detect(text) == []
    assert not any(s.kind == 'quran' for s in arabic.spans(text, quran=index))


def test_a4_strict_keeps_supplied_index(index):
    text = 'گفت: هذا باب جميل'
    assert [s for s in arabic.spans(text, quran=index, strict=True) if s.kind == 'quran'] == index.detect(text)
    assert sum(arabic.mask(text, quran=index, strict=True)) >= 3


@pytest.mark.parametrize('text, expected', [
    ('این ‌ کتاب است', ['این', 'کتاب', 'است']),
    ('إن ‌ هذا باب جميل', ['إن', 'هذا', 'باب', 'جميل']),
    ('ً ‌ ‍ ـ کتاب‌خانه ‌کتاب کتاب‌', ['کتاب‌خانه', 'کتاب', 'کتاب']),
    ('علم‌دار و مُعَلِّم', ['علم‌دار', 'و', 'مُعَلِّم']),
    ('ک‍تاب', ['ک‍تاب']),
    ('بالْعَد‌‌ْلِ', ['بالْعَد‌‌ْلِ']),
])
def test_a5_words_require_letters_and_internal_joiners(text, expected):
    tokens = arabic.TOKEN.findall(text)
    assert tokens == expected
    words = arabic.words(text)
    assert [text[w.start:w.end] for w in words] == expected
    assert len(arabic.mask(text)) == len(expected)


@pytest.mark.parametrize('expression', ['مع الوصف', 'مع‌الوصف'])
def test_a6_persian_loan_expression_abstains(expression):
    text = f'این {expression} روشن است'
    assert not arabic.spans(text)
    assert not any(arabic.mask(text))


def test_a6_singleton_decisions_agree_across_apis():
    model = WordModel({'words': {}, 'grams': {}, 'trees': [], 'threshold': .65})
    model.probability = lambda row: .99 if row[6] == 2 / 6 else .01
    text = 'این مع روشن است'
    assert not any(w.arabic for w in model.words(text))
    assert model.detect(text) == []


def test_normalizer_mode_requires_rule_agreement_without_an_index():
    model = WordModel({'words': {}, 'grams': {}, 'trees': [], 'threshold': .65})
    model.probability = lambda row: .99999
    text = 'این خانه روشن است'
    assert model.detect(text)
    assert model.detect(text, strict=True) == []


def test_calibration_and_threshold_override():
    model = WordModel({'words': {}, 'grams': {}, 'trees': [], 'bias': 2,
                       'calibration': [2, -4], 'threshold': .6})
    assert model.probability([]) == .5
    assert model.detect('الف ب') == []
    assert len(model.detect('الف ب', threshold=.5)) == 1
    for threshold in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            model.words('الف ب', threshold=threshold)


def test_model_only_runs_do_not_bridge_foreign_words():
    model = WordModel({'words': {}, 'grams': {}, 'trees': [], 'bias': 10})
    text = 'الف ب слово ج د'
    assert [text[s.start:s.end] for s in model.detect(text)] == ['الف ب', 'ج د']


def test_verified_and_model_intervals_remain_separate(index):
    model = WordModel({'words': {}, 'grams': {}, 'trees': [], 'bias': 10})
    text = 'الف ب هذا باب جميل ج د'
    found = model.detect(text, quran=index)
    assert [(text[s.start:s.end], s.kind) for s in found] == [
        ('الف ب', 'passage'), ('هذا باب جميل', 'quran'), ('ج د', 'passage')]


def test_a7_all_category_is_counted_once():
    row = {'text': 'الف ب', 'spans': [[0, 5, 'passage']], 'category': 'all'}
    counts = annotated([row], lambda text: [arabic.ArabicSpan(0, 5, ())])['categories']['all']
    assert (counts['tp'], counts['span_tp'], counts['lines']) == (2, 1, 1)


def test_pool_weighted_counts_and_false_alarm_denominator():
    rows = [{'text': 'الف ب', 'spans': [], 'category': 'all', 'weight': 9},
            {'text': 'الف ب', 'spans': [[0, 5, 'passage']], 'weight': 1}]
    counts = annotated(rows, lambda text: [arabic.ArabicSpan(0, 5, ())])['categories']['all']
    assert counts['precision'] == .1 and counts['recall'] == 1
    assert counts['line_recall'] == 1 and counts['lines'] == 10
    assert counts['false_alarms_per_1000_lines'] == 900
