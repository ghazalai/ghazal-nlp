import pytest

from ghazal_nlp import vowels as v, g2p_eval as g, vowel_context as c


@pytest.fixture
def model():
    return {"words": {"گل": {"forms": ["گُل", "گِل"], "bias": [0, 0],
                            "weights": {"1:زیباست": [7, -7], "1:رس": [-7, 7]}}}}


@pytest.mark.parametrize("word,sound,form", [
    ("کرد", "kard", "کَرْد"), ("کرد", "kord", "کُرْد"),
    ("ملک", "malek", "مَلِک"), ("ملک", "molk", "مُلْک"),
    ("ملک", "melk", "مِلْک"), ("ملک", "malak", "مَلَک"),
    ("کتاب", "ketAb-e", "کِتَاب"), ("خانه", "xAne-ye", "خَانِه"),
    ("رب", "rabb", "رَبّ"), ("خواب", "xAb", "خَواب"),
    ("این", "in", "اِین"), ("آب", "Ab", "آب"),
])
def test_alignment(word, sound, form):
    assert g.vowel_form(word, sound) == form


def test_alignment_rejects_missing_consonants_and_vowels():
    assert g.vowel_form("کرد", "krd") == "کْرْد"
    assert g.vowel_form("کرد", "kat") is None
    assert g.vowel_form("گل", "gollx") is None
    assert list(g.vowel_alignment("گل گل", ["gol", "gel"])) == []


def test_context_and_abstention(model):
    assert v.mark("گل زیباست", context=model, ezafe=False) == "گُل زیباست"
    assert v.mark("گل رس", context=model, ezafe=False) == "گِل رس"
    assert v.mark("گل", context=model, ezafe=False) == "گل"
    assert v.mark("xyz", context=model, ezafe=False) == "xyz"


def test_original_spelling_whitespace_and_marks(model):
    assert v.mark("\tگل  زیباست!\n", context=model, ezafe=False) == "\tگُل  زیباست!\n"
    assert v.mark("گِل زیباست", context=model, ezafe=False) == "گِل زیباست"
    marked=v.mark("گل رس", context=model, ezafe=False)
    assert v.mark(marked, context=model, ezafe=False) == marked


def test_full_and_minimal():
    model={"words":{"کرد":{"forms":["کَرْد","کُرْد"],"bias":[10,0],"weights":{}},
                    "کتاب":{"forms":["کِتاب"],"bias":[0],"weights":{},"reliability":1}}}
    assert v.mark("کرد کتاب", mode="full",context=model,lexicon=v.Lexicon(),ezafe=False)=="کَرد کِتاب"
    assert v.mark("کرد کتاب", mode="minimal",context=model,lexicon=v.Lexicon(),ezafe=False)=="کَرد کتاب"


def test_untrained_lexicon_cannot_claim_certainty():
    lex=v.build_lexicon({"گُل":50})
    assert v.mark("گل",context={"words":{}},lexicon=lex,ezafe=False)=="گل"


def test_lexical_calibration_belongs_to_a_form_not_just_a_word(monkeypatch):
    data={'words':{},'lexical_reliability':{'گل':{'گُل':.95}}}
    lex=v.build_lexicon({'گُل':50})
    monkeypatch.setattr(v,'lexicon',lambda:lex)
    assert v.choose('گل',context=data)[0].confidence==.95
    supplied=v.build_lexicon({'گِل':50})
    choice=v.choose('گل',context=data,lexicon=supplied)[0]
    assert choice.form=='گِل' and choice.confidence==0
    assert v.mark('گل',mode='full',context=data,lexicon=supplied,ezafe=False)=='گل'


def test_maddah_already_writes_its_vowel():
    model={"words":{"آب":{"forms":["آَب"],"bias":[0],"weights":{},"reliability":1}}}
    assert v.mark("آب",mode="full",context=model,ezafe=False)=="آب"


def test_bad_options(model):
    with pytest.raises(ValueError):v.mark("گل",mode="other",context=model)
    with pytest.raises(ValueError):v.mark("گل",threshold=2,context=model)
    with pytest.raises(ValueError):v.choose("گل",profile="other",context=model)


def test_metre_veto_does_not_inflate_confidence(monkeypatch, model):
    from types import SimpleNamespace
    monkeypatch.setattr(c,"_scanner",lambda _:SimpleNamespace(
        scan=lambda text:SimpleNamespace(metres=[] if "گُل" in text else ["m"],licences=0)))
    choice=v.choose("گل زیباست",context=model,profile="verse",metre="m")[0]
    assert choice.form=="گِل" and choice.reason=="metre" and choice.confidence<0.01


def test_ezafe_uses_its_own_api(monkeypatch, model):
    from ghazal_nlp import ezafe
    monkeypatch.setattr(ezafe,"junctions",lambda *args,**kwargs:[
        ezafe.Junction("خانه","پدر",(0,8),"unmarked","ezafe","model")])
    assert v.mark("خانه پدر",context=model)=="خانه پدر"
    assert v.mark("خانه پدر",context=model,ezafe=True)=="خانهٔ پدر"


def test_single_retained_candidate_is_not_automatically_certain():
    model={"words":{"این":{"forms":["اَیْن"],"bias":[0],"weights":{}}}}
    assert v.choose("این", context=model)[0].confidence == 0
    assert v.mark("این", mode="full", context=model, ezafe=False) == "این"


def test_shadda_is_added_independently_of_existing_vowels():
    model={"words":{"مدرس":{"forms":["مُدَرِّس"],"bias":[0],"weights":{},"reliability":1}}}
    for text in ("مدرس", "مُدَرِس"):
        assert v.mark(text, mode="full", context=model, ezafe=False) == "مُدَرِّس"


def test_existing_duplicates_order_and_tatweel_are_preserved(model):
    assert v.mark("گُُل زیباست", context=model, ezafe=False) == "گُُل زیباست"
    assert v.mark("گُّل زیباست", context=model, ezafe=False) == "گُّل زیباست"
    assert v.mark("گـل زیباست", context=model, ezafe=False) == "گُـل زیباست"


def test_supplied_lexicon_augments_trained_candidates(model):
    lex=v.build_lexicon({"گَل":5})
    choice=v.choose("گَل", context=model, lexicon=lex)[0]
    assert choice.form == "گَل" and choice.reason == "lexicon" and choice.confidence == 0
    assert set(choice.readings) == {"گَل", "گُل", "گِل"}


def test_dictionary_prior_cannot_replace_an_ambiguous_context_choice(model):
    model['fallback']={'گل':{'forms':['گَل'],'counts':[20]}}
    assert v.choose('گَل',context=model)[0].confidence==0
    model['fallback']['گل']['reliability']=.95
    choice=v.choose('گل',context=model)[0]
    assert choice.form in ('گُل','گِل') and choice.reason=='context'
    assert v.choose('گَل',context=model)[0].confidence==.95
    assert v.choose('گُل',context=model)[0].form=='گُل'


def test_shared_aligner_matches_dictionary_and_sentence_adapters():
    from ghazal_nlp import vowel_align as a
    for word, sound in [("کتاب","ketAb"),("خانه","xAne"),("خواب","xAb"),("آب","Ab"),("این","in")]:
        assert a.align_sound(word,sound) == g.vowel_form(word,sound)
    assert a.align_sound("کتاب","ketab","homorich") == "کِتَاب"
    assert a.align_sound("کتاب","/keˈtɒːb/","ipa") == "کِتَاب"
    assert a.align_letters("تیره","[ رَ / رِ ]") == ("تیرَه", "تیرِه")
    assert not a.align_letters("تت", "[ تَ ]")


@pytest.mark.parametrize("word,sound,form", [
    ("رو", "ro", "رُو"), ("رو", "ru", "رُو"), ("دوم", "dovom", "دُوُم"),
    ("اول", "avval", "اَوَّل"), ("سیاره", "sayyAre", "سَیَّارِه"),
    ("اول", "avvval", None), ("بیا", "biyA", "بِیَا"), ("دیدم", "didam", "دِیدَم"),
    ("خواند", "xAnd", "خَوانْد"), ("خواب", "xvAb", "خْوَاب"),
    ("خوب", "xAb", None), ("خواوب", "xAb", None), ("دوان", "dAn", None),
])
def test_shared_carriers_and_consonantal_gemination(word, sound, form):
    from ghazal_nlp import vowel_align as a
    assert a.align_sound(word, sound) == g.vowel_form(word, sound) == form


def test_minimal_ambiguity_gate_is_separate_from_confidence(model):
    model['words']['گل']['counts']=[99,1]
    assert v.mark("گل زیباست",context=model,ambiguity=0,ezafe=False) == "گُل زیباست"
    assert v.mark("گل زیباست",context=model,ambiguity=.05,ezafe=False) == "گل زیباست"


def test_public_signatures_expose_marking_options():
    import inspect
    assert {'mode','threshold','ambiguity','lexicon','context','profile','metre','ezafe'} <= set(inspect.signature(v.mark).parameters)
    assert {'lexicon','context','profile','metre'} <= set(inspect.signature(v.choose).parameters)
