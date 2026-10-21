import glob

import pytest

from ghazal_nlp import verb_eval
from ghazal_nlp.verb import ZWNJ, Analyzer, Lexicon, Reading, Verb, analyzer, constructions

A = analyzer()


def lemmas(token):
    return {r.lemma for r in A.analyze(token).readings}


@pytest.mark.parametrize("token,lemma,features", [
    ("می‌روم", "رفتن", "Tense=Pres|Aspect=Imp|Mood=Indicative|Person=1|Number=Sing|Polarity=Pos"),
    ("نمی‌گوید", "گفتن", "Tense=Pres|Aspect=Imp|Mood=Indicative|Person=3|Number=Sing|Polarity=Neg"),
    ("رفت", "رفتن", "Tense=Past|Aspect=Perf|Mood=Indicative|Person=3|Number=Sing|Polarity=Pos"),
    ("نرفتم", "رفتن", "Tense=Past|Aspect=Perf|Mood=Indicative|Person=1|Number=Sing|Polarity=Neg"),
    ("بروم", "رفتن", "Tense=Pres|Mood=Subjunctive|Person=1|Number=Sing|Polarity=Pos"),
    ("رفته‌ام", "رفتن", "Tense=Past|Aspect=Perf|Mood=Indicative|Person=1|Number=Sing|Polarity=Pos"),
])
def test_readings(token, lemma, features):
    assert any(r.lemma == lemma and r.features() == features for r in A.analyze(token).readings)


def test_what_it_refuses():
    assert all(r.person == 3 and r.number == "sing" for r in A.analyze("می‌گوید").readings)
    assert {"شدن", "شستن"} <= lemmas("می‌شود")
    assert A.analyze("کتاب").abstention == "unknown_stem"
    assert A.analyze("").abstention == "empty"
    assert A.analyze("می‌توان").readings and A.analyze("نمی‌توان").readings
    assert not A.analyze("توان").readings and not A.analyze("می‌نوش").readings
    assert not A.analyze("می‌آرد").readings and not A.analyze("آرد").readings and "آوردن" in lemmas("می‌آورد")
    for form, lemma in (("می‌باید", "بایستن"), ("می‌نهد", "نهادن"), ("می‌بارد", "باریدن"), ("می‌چکد", "چکیدن")):
        assert lemma in lemmas(form)


@pytest.mark.parametrize("infinitive,kwargs,expected", [
    ("رفتن", dict(person=1), "می‌روم"),
    ("رفتن", dict(tense="past", number="plur"), "رفتند"),
    ("رفتن", dict(tense="past", person=1, aspect="imperfective"), "می‌رفتم"),
    ("گفتن", dict(), "می‌گوید"),
    ("گفتن", dict(mood="imperative", person=2), "بگو"),
    ("دیدن", dict(polarity="neg"), "نمی‌بیند"),
    ("برداشتن", {}, "برمی‌دارد"),
    ("بازگشتن", dict(tense="past", number="plur"), "بازگشتند"),
    ("برداشتن", dict(tense="past", polarity="neg"), "برنداشت"),
    ("بازگشتن", dict(mood="subjunctive", polarity="neg"), "بازنگردد"),
    ("برداشتن", dict(mood="imperative", person=2), "بردار"),
    ("کردن", dict(tense="past", aspect="imperfective_perfect", person=1), "می‌کرده‌ام"),
])
def test_conjugation_round_trips(infinitive, kwargs, expected):
    assert A.conjugate(infinitive, **kwargs) == expected
    assert infinitive in lemmas(expected)


def test_generation():
    assert A.conjugate("رفتن", person=4) is None
    assert A.conjugate("گفتن", tense="past", mood="subjunctive") is None
    assert A.conjugate("گفتن", mood="imperative", person=3) is None
    assert A.conjugate("رفتن", aspect="imperfective_perfect") is None
    assert A.conjugate("گفتن", mood="subjunctive", aspect="imperfective") is None
    for person in (1, 2, 3):
        for number in ("sing", "plur"):
            form = A.conjugate("رفتن", tense="past", person=person, number=number)
            assert any(r.lemma == "رفتن" and (r.person, r.number) == (person, number) for r in A.analyze(form).readings)


@pytest.mark.parametrize("token,lemma,tense,polarity", [
    ("برمی‌دارد", "برداشتن", "present", "pos"), ("می‌بردارد", "برداشتن", "present", "pos"),
    ("درنمی‌آورد", "درآوردن", "past", "neg"), ("نمی‌درآورد", "درآوردن", "past", "neg"),
    ("بازمی‌گشت", "بازگشتن", "past", "pos"), ("می‌بازگردد", "بازگشتن", "present", "pos"),
    ("برداشت", "برداشتن", "past", "pos"), ("بازگشتند", "بازگشتن", "past", "pos"),
    ("برنداشت", "برداشتن", "past", "neg"), ("بازنگردد", "بازگشتن", "present", "neg"),
    ("درمی‌گذرد", "درگذشتن", "present", "pos"), ("برمی‌خورد", "برخوردن", "present", "pos"),
    ("سرمی‌برد", "سربریدن", "present", "pos"), ("فرو‌رفت", "فرورفتن", "past", "pos"),
])
def test_preverbs(token, lemma, tense, polarity):
    rs = [r for r in A.analyze(token).readings if (r.lemma, r.tense, r.polarity) == (lemma, tense, polarity)]
    assert rs and all(r.preverb and r.base_lemma and r.rules[-1].startswith("separable_preverb") for r in rs)


def test_preverb_ambiguity_and_omitted_b():
    assert {r.tense for r in A.analyze("درمی‌آورد").readings if r.lemma == "درآوردن"} == {"past", "present"}
    assert lemmas("بازمی‌گردد") == {"بازگردیدن", "بازگشتن"}
    assert any(r.mood == "imperative" and r.lemma == "برداشتن" for r in A.analyze("بردار").readings)
    assert any(r.mood == "subjunctive" and r.lemma == "برداشتن" for r in A.analyze("بردارد").readings)


@pytest.mark.parametrize("token", [
    "در", "بر", "در خانه", "بر میز", "در می‌خواند", "بر می‌خواند", "درکشتن", "درداشتن", "سرکشتن", "بازکشتن",
    "درمی‌خواند", "برمی‌پژوهد", "وامی‌رود", "برمی‌ناشناختهد", "می‌ناشناخته", "نمی‌ناشناخته",
    "بایدن", "درایدن", "می‌بایدن", "درمی‌بایدن", "بر‌می‌‌دارد", "بر می‌دارد",
])
def test_abstains(token):
    assert not A.analyze(token).readings


@pytest.mark.parametrize("token", ["دریا", "دریای", "برفت", "برفتند", "دریاورید", "برنداز"])
def test_a_preverb_does_not_license_prefix_alternates(token):
    assert not any(r.preverb for r in A.analyze(token).readings)


@pytest.mark.parametrize("token,lemma", [
    ("بیا", "آمدن"), ("نیا", "آمدن"), ("درنیا", "درآمدن"), ("درنیاورید", "درآوردن"), ("دربیاورید", "درآوردن"),
    ("برنفتد", "برافتادن"), ("برمی‌فتد", "برافتادن"), ("می‌برفتد", "برافتادن"),
    ("بازداشت", "بازداشتن"), ("دریافت", "دریافتن"), ("درخواست", "درخواستن"), ("درآمد", "درآمدن"),
    ("بازگشت", "بازگشتن"), ("درگذشت", "درگذشتن"), ("درمان", "درماندن"),
])
def test_prefixed_alternates_and_nominal_homographs(token, lemma):
    assert lemma in lemmas(token)


def test_alternates_need_a_prefix():
    simple = Analyzer(preverbs=())
    assert not simple.analyze("یای").readings and not simple.analyze("فتند").readings
    assert all(simple.analyze(t).readings for t in ("نیای", "نفتند", "گویید", "هستم"))
    assert Analyzer(Lexicon([Verb("test", ("test-past",), ("یا",))]), preverbs=()).analyze("یای").readings


@pytest.mark.parametrize("token,lemma", [
    ("بروید", "رفتن"), ("نروید", "رفتن"), ("بگویید", "گفتن"), ("برآورید", "برآوردن"),
    ("بردارید", "برداشتن"), ("درنیاورید", "درآوردن"), ("بازنگردید", "بازگشتن"),
])
def test_plural_imperative(token, lemma):
    rs = [r for r in A.analyze(token).readings if r.lemma == lemma and (r.person, r.number) == (2, "plur")]
    assert {r.mood for r in rs} >= {"imperative", "subjunctive"} and all(r.tense == "present" for r in rs)
    assert any("imperative" in r.rules for r in rs if r.mood == "imperative")


@pytest.mark.parametrize("token", ["می‌روید", "نمی‌روید", "برمی‌آورید", "نمی‌برآورید", "بگوید"])
def test_no_plural_imperative_after_mi_or_glide(token):
    assert not any(r.mood == "imperative" for r in A.analyze(token).readings)


@pytest.mark.parametrize("token,lemma,polarity", [
    ("رفته", "رفتن", "pos"), ("دانسته", "دانستن", "pos"), ("نرفته", "رفتن", "neg"), ("ندانسته", "دانستن", "neg"),
    ("گفته", "گفتن", "pos"), ("نگفته", "گفتن", "neg"), ("نوشته", "نوشتن", "pos"), ("بسته", "بستن", "pos"),
    ("خندیده", "خندیدن", "pos"), ("خوانده", "خواندن", "pos"), ("برداشته", "برداشتن", "pos"),
    ("برنداشته", "برداشتن", "neg"), ("فروریخته", "فروریختن", "pos"), ("فرونریخته", "فروریختن", "neg"),
    ("فرو‌ریخته", "فروریختن", "pos"),
])
def test_participles_leave_agreement_open(token, lemma, polarity):
    r, = [r for r in A.analyze(token).readings if r.lemma == lemma and r.verb_form == "participle"]
    assert (r.stem_type, r.tense, r.aspect, r.polarity, r.person, r.number) == ("past", "past", "perfect", polarity, None, None)
    assert "VerbForm=Part" in r.features() and "Person=" not in r.features() and "past_participle" in r.rules


@pytest.mark.parametrize("token,lemma,polarity", [
    ("می‌دانسته", "دانستن", "pos"), ("می‌بوده", "بودن", "pos"), ("نمی‌کرده", "کردن", "neg"),
    ("می‌شده", "شدن", "pos"), ("بازمی‌گشته", "بازگشتن", "pos"), ("می‌بازگشته", "بازگشتن", "pos"),
])
def test_mi_participles(token, lemma, polarity):
    r = next(r for r in A.analyze(token).readings if r.lemma == lemma)
    assert (r.aspect, r.verb_form, r.polarity, r.person, r.number) == ("imperfective_perfect", "participle", polarity, None, None)
    assert "Construction=ImperfectivePerfect" in r.features() and "mi_participle" in r.rules


@pytest.mark.parametrize("token", ["خانه", "خنده", "ناشناخته‌ه", "گفتهه", "بگفته", "بنرفته", "برکشته", "سرکشته", "بر داشته"])
def test_no_participle_without_a_known_past_stem(token):
    assert not any(r.verb_form == "participle" for r in A.analyze(token).readings)


def test_participles_keep_their_homographs():
    rs = A.analyze("برده").readings
    assert any(r.lemma == "بردن" and r.verb_form == "participle" for r in rs)
    assert any(r.lemma == "بردادن" and r.mood == "imperative" for r in rs)
    rs = Analyzer(Lexicon([Verb("part", ("آزمود",), ("آزما",)), Verb("finite", ("آزموده",), ("تست",)),
                           Verb("imperative", ("آزمودید",), ("آزموده",))]), preverbs=()).analyze("نآزموده").readings
    assert {"part", "finite", "imperative"} <= {r.lemma for r in rs}
    rs = Analyzer(Lexicon([Verb("الف", ("آزمود",), ("آزما",)), Verb("بیتا", ("آزموده",), ("تست",))]), preverbs=()).analyze("می‌آزموده").readings
    assert {r.verb_form for r in rs} == {"participle", None}
    assert not Analyzer(Lexicon(), preverbs=()).analyze("رفته").readings
    assert not Analyzer(preverbs=()).analyze("برداشته").readings


def test_perfects():
    r = next(r for r in A.analyze("نمی‌کرده‌ایم").readings if r.lemma == "کردن")
    assert (r.aspect, r.person, r.number, r.verb_form) == ("imperfective_perfect", 1, "plur", "finite")
    assert A.analyze("کرده‌ایم").readings[0].aspect == "perfect"
    for token, aspect, person, number in (("می‌رفته", "imperfective_perfect", None, None), ("رفته‌ام", "perfect", 1, "sing"),
                                          ("برنداشته‌اید", "perfect", 2, "plur")):
        assert all((r.aspect, r.person, r.number) == (aspect, person, number) for r in A.analyze(token).readings)


@pytest.mark.parametrize("token", ["برفت", "بگفت", "بدید", "بهشت", "بکرد"])
def test_no_b_past(token):
    assert not any(r.tense == "past" and r.verb_form == "finite" and r.stem != token for r in A.analyze(token).readings)


@pytest.mark.parametrize("text,kind,aspect,mood,person", [
    ("می‌گردیده باش", "perfect_imperative", "imperfective_perfect", "imperative", 2),
    ("می‌بوده باشد", "perfect_subjunctive", "imperfective_perfect", "subjunctive", 3),
    ("رفته بود", "past_perfect", "perfect", "indicative", 3),
    ("رفته باشد", "perfect_subjunctive", "perfect", "subjunctive", 3),
    ("رفته است", "perfect", "perfect", "indicative", 3),
    ("خواهد رفت", "future", None, "indicative", 3),
    ("گفته شد", "passive", "perfective", "indicative", 3),
    ("گفته می‌شد", "passive", "imperfective", "indicative", 3),
    ("نرفته بودم", "past_perfect", "perfect", "indicative", 1),
    ("گفته باشند", "perfect_subjunctive", "perfect", "subjunctive", 3),
    ("خواهم بازگشت", "future", None, "indicative", 1),
])
def test_constructions(text, kind, aspect, mood, person):
    r = next(r for r in constructions(*(A.analyze(t) for t in text.split())) if r.construction == kind)
    assert (r.aspect, r.mood, r.person, r.voice) == (aspect, mood, person, "passive" if kind == "passive" else "active")


def test_construction_limits():
    assert not A.analyze("باش").readings and not A.analyze("است").readings
    assert {r.mood for r in A.analyze("باشد").readings} == {"indicative", "subjunctive"}
    assert not constructions(A.analyze("رفته‌است"), A.analyze("بود"))
    assert {r.mood for r in constructions(A.analyze("رفته"), A.analyze("باشید"))} == {"subjunctive", "imperative"}
    r, = constructions(A.analyze("نخواهیم"), A.analyze("رفت"))
    assert (r.polarity, r.person, r.number, r.components[0].lemma, r.components[1].lemma) == ("neg", 1, "plur", "خواستن", "رفتن")
    r, = constructions(A.analyze("نرفته"), A.analyze("نباشد"))
    assert r.component_polarities == ("neg", "neg")
    for text in ("خانه است", "رفته مردم", "خواهد رفتند", "خواهد برفت", "می‌خواهد رفت", "خواهد نرفت", "خواسته رفت", "فرونشسته بود"):
        assert not constructions(*(A.analyze(t) for t in text.split()))
    for separator in ("", "، ", ". ", " و "):
        assert not constructions(A.analyze("رفته"), A.analyze("بود"), separator=separator)


def test_inventory():
    assert len(A.lexicon) == 251 and "برداشتن" not in A.lexicon.by_infinitive
    r = Reading("رفت", "رفتن", "رفت", "past", "past", "perfective", "indicative", 3, "sing", "pos")
    assert r.features() == "Tense=Past|Aspect=Perf|Mood=Indicative|Person=3|Number=Sing|Polarity=Pos"


def test_ud_scoring(tmp_path):
    path = tmp_path / "t.conllu"
    path.write_text("1\tمی‌روم\tرفت#رو\tVERB\t_\tNumber=Sing|Person=1|Tense=Pres\t0\troot\t_\t_\n"
                    "2\tکتاب\tکتاب\tNOUN\t_\t_\t1\tobj\t_\t_\n3\tاست\tاست\tAUX\t_\tTense=Pres\t1\tcop\t_\t_\n", encoding="utf-8")
    r = verb_eval.evaluate([path], A)
    assert (r["verbs"], r["analysed"], r["lemma_among_readings"], r["features_when_analysed"]) == (2, 0.5, 0.5, 1.0)
    assert r["mi_words"] == 1 and r["mi_analysed"] == 1.0


@pytest.mark.parametrize("form,lemma,person,number", [
    ("بیاید", "آمدن", 3, "sing"), ("بیایم", "آمدن", 1, "sing"), ("بیایید", "آمدن", 2, "plur"), ("نیامد", "آمدن", 3, "sing"),
    ("بیفتد", "افتادن", 3, "sing"), ("نینداخت", "انداختن", 3, "sing"), ("بیاورد", "آوردن", 3, "sing"),
    ("درنیامد", "درآمدن", 3, "sing"),
])
def test_the_glide_after_b_and_n(form, lemma, person, number):
    assert (lemma, person, number) in {(r.lemma, r.person, r.number) for r in A.analyze(form).readings}
    assert not any(r.person == 2 and r.number == "plur" for r in A.analyze("بیاید").readings)


@pytest.mark.parametrize("infinitive,kwargs,expected", [
    ("آمدن", {"mood": "subjunctive", "person": 1}, "بیایم"), ("آمدن", {"mood": "imperative", "person": 2}, "بیا"),
    ("آمدن", {"mood": "imperative", "person": 2, "number": "plur"}, "بیایید"),
    ("انداختن", {"mood": "imperative", "person": 2}, "بینداز"), ("آمدن", {"tense": "past", "polarity": "neg"}, "نیامد"),
])
def test_generation_writes_the_glide(infinitive, kwargs, expected):
    assert A.conjugate(infinitive, **kwargs) == expected


@pytest.mark.parametrize("written,lemma", [
    ("میرود", "رفتن"), ("نمیخواهد", "خواستن"), ("دانستهاند", "دانستن"), ("مي‌رود", "رفتن"), ("ﻣﻲ‌رود", "رفتن"),
])
def test_other_encodings_and_a_missing_zwnj(written, lemma):
    assert lemma in lemmas(written)
