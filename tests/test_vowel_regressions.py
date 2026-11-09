import pytest

from ghazal_nlp import vowels as v, vowel_context as c


COMMON = '''گل ملک کرد کرم سر مهر شکر بر در علم حلم حسن حکم خلق برد مرد پر دم
کشت کشتی ترک ده سحر دور سیر دین عقد عرف عرض نقل قصر سفر عمر حمل نفس قدم قسم
صفر محرم مدرس معین مسلم ببر برنده بکش ببرد می‌برد می‌کشد درد رمان'''.split()


@pytest.mark.parametrize('word', COMMON)
def test_common_homographs_have_marked_source_readings(word):
    readings = v.lookup(word)
    assert len(readings) >= 2, word
    assert all(v.is_marked(r.form) and r.sources for r in readings)


@pytest.mark.parametrize(('form', 'written'), [
    ('رَا', 'را'), ('اِین', 'این'), ('بُود', 'بود'),
    ('مِی‌سَازَد', 'می‌سازَد'), ('دَوْر', 'دَوْر'),
    ('دَوَران', 'دَوَران'), ('تَیَمُّم', 'تَیَمُّم'), ('مَثَلاً', 'مَثَلاً'),
    ('خَوانْد', 'خوانْد'), ('دُوْر', 'دَوْر'), ('سِیْر', 'سَیْر'),
    ('هٰذا', 'هذا'), ('سُوَار', 'سُوار'), ('آبْسُوَار', 'آبْسُوار'), ('خِیَالْبَافِی', 'خِیالْبافی'),
    ('آبَادِیَان', 'آبادِیان'), ('آدِیُو', 'آدِیو'), ('آلُومِینِیُوم', 'آلومینِیوم'), ('مُوَافِقِید', 'مُوافِقید'),
    ('پِیوَاز', 'پیواز'), ('گُویِی', 'گویی'), ('مِیآوَرْد', 'میآوَرْد'), ('وَشَهِدُوا', 'وَشَهِدوا'),
    ('اَبُوالْغازی', 'اَبوالْغازی'), ('زِیبَا', 'زیبا'),
])
def test_editorial_long_vowels(form, written):
    assert v.editorial(form) == v.canonicalize(written)


def test_rare_readings_and_function_words_do_not_trigger_minimal_marks():
    record = {'forms':['بِه','بَه'], 'counts':[996,4]}
    assert c.alternatives({}, 'به', record, .05, 'بِه') == ()
    assert c.alternatives({}, 'به', record, .05, 'بَه') == ()
    record = {'forms':['گُل','گِل'], 'counts':[60,40]}
    assert c.alternatives({}, 'گل', record, .05, 'گُل') == ('گِل',)


def test_meaning_probability_combines_pronunciation_variants():
    data = {'homographs':{'شکر':{'groups':[['شِکَر','شَکَر'],['شُکر']]}}}
    rec = {'forms':['شِکَر','شَکَر','شُکْر']}
    best, p = c.rank(data,'شکر',rec,[0,0,.2],range(3))
    assert best in (0,1) and p > .6


def test_long_vowel_rendering_preserves_existing_marks():
    assert c._overlay('رَا', v.editorial('رَا')) == 'رَا'
    assert c._overlay('می‌سازد', v.editorial('مِی‌سَازَد')) == 'می‌سازَد'


def test_minimal_distinguishes_a_diphthong_from_a_long_vowel():
    assert c.minimal(v.editorial('دُوْر'),(v.editorial('دُور'),)) == 'دَور'
    assert c.minimal(v.editorial('قُوَّت'),(v.editorial('قُوت'),)) == 'قوّت'


def test_editorial_is_idempotent_over_the_lexicon():
    forms = {r.form for e in v.lexicon().values() for r in e.readings}
    assert all(v.editorial(f) == v.editorial(v.editorial(f)) for f in forms)


def test_choose_spells_as_full_mode():
    assert [x.form for x in v.choose('او به خانه رفت')] == ['او', 'بِه', 'خانِه', 'رَفت']
    assert ' '.join(x.form for x in v.choose('ملک را به ملک بخشید')) == v.mark('ملک را به ملک بخشید', mode='full')


def test_minimal_prefers_short_vowels_to_sukun():
    assert c.minimal('کِرْم',('کَرَم','کُرُم')) == 'کِرم'
    assert c.minimal('کِرْم',('کَرَم','کِرِم')) == 'کِرْم'
    assert c.minimal('کِرْم',('کَرَم','کِرِم'),{'کِرم'}) == 'کِرم'


def test_minimal_does_not_emit_an_incomplete_disambiguation():
    assert c.minimal('گُل',('گُّل','گِل')) == 'گل'


def test_written_diphthongs_constrain_phonetic_candidates():
    model={'words':{'دور':{'forms':['دُور','دُوْر'],'bias':[5,0],'weights':{}}}}
    assert v.choose('دَور',context=model,lexicon=v.Lexicon())[0].form=='دَور'


def test_real_function_word_sentences_stay_unmarked():
    assert v.mark('او در خانه بود و به مدرسه رفت', ezafe=False) == 'او در خانه بود و به مدرسه رفت'


def test_ethnic_coordination_and_biological_compounds():
    assert v.mark('او از کرد و ترک بود') == 'او از کُرد و تُرک بود'
    choice=v.choose('کرم ابریشم پیله می‌سازد')[0]
    assert c.signature(choice.form)==c.signature('کِرْم')
    assert v.mark('کرم ابریشم پیله می‌سازد')=='کِرم ابریشم پیله می‌سازد'


def test_transferred_property_and_its_recipient_have_distinct_readings():
    choices=[x for x in v.choose('ملک را به ملک بخشید') if x.word=='ملک']
    assert [c.signature(x.form) for x in choices]==[c.signature('مِلْک'),c.signature('مَلِک')]
    assert v.mark('ملک را به ملک بخشید')=='مِلک را به ملِک بخشید'
