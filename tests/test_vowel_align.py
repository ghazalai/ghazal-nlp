from ghazal_nlp import vowel_align as a, vowels as v


def test_letter_cues_do_not_guess_missing_letters():
    assert a.align_letters("تهییل", "[ تَ ]") == ("تَهییل",)
    assert a.align_letters("تیره", "[ رَ / رِ ]") == ("تیرَه", "تیرِه")
    assert not a.align_letters("تت", "[ تَ ]")
    assert not a.align_letters("تفأل", "[ تَ فَ ءْ ءُ ]")
    assert not a.align_letters("تنخواه", "[ تَ خوا / خا ]")
    assert not a.align_letters("کتاب", "[ ـَ ]")


def test_sound_alignment_checks_every_letter():
    assert a.align_sound("کتاب", "ketAb") == "کِتَاب"
    assert a.align_sound("گل", "gol") == "گُل"
    assert a.align_sound("خانه", "xAne") == "خَانِه"
    assert a.align_sound("راه", "rAh") == "رَاه"
    assert a.align_sound("رو", "ro") == "رُو"
    assert a.align_sound("رو", "ru") == "رُو"
    assert a.align_sound("دوم", "dovom") == "دُوُم"
    assert a.align_sound("ده", "deh") == "دِه"
    assert a.align_sound("ده", "de") == "دِه"
    assert a.align_sound("شکر", "Sekar") == "شِکَر"
    assert a.align_sound("کتاب", "gol") is None
    assert a.align_sound("کتاب", "ketab", "homorich") == "کِتَاب"
    assert a.align_sound("کتاب", "kitāb", "roman") == "کِتَاب"
    assert a.align_sound("کتاب", "/keˈtɒːb/", "ipa") == "کِتَاب"
    assert a.align_sound("جنگ", "/d͡ʒæŋɡ/", "ipa") is None
    assert a.align_sound("جنگ", "/d͡ʒæng/", "ipa") == "جَنْگ"
    assert a.align_sound("کتاب", "китоб", "tajik") == "کِتَاب"
    assert a.align_sound("روز", "рӯз", "tajik") is None


def test_silent_waw_is_bounded_to_khwa():
    assert a.align_sound("خواب", "xAb") == "خَواب"
    assert a.align_sound("خواند", "xAnd") == "خَوانْد"
    assert a.align_sound("خواب", "xvAb") == "خْوَاب"
    assert a.align_sound("خوب", "xAb") is None
    assert a.align_sound("خواوب", "xAb") is None
    assert a.align_sound("دوان", "dAn") is None


def test_consonantal_waw_and_yeh_can_geminate():
    assert a.align_sound("اول", "avval") == "اَوَّل"
    assert a.align_sound("سیاره", "sayyAre") == "سَیَّارِه"
    assert a.align_sound("اول", "avvval") is None
    assert a.align_sound("بیا", "biyA") == "بِیَا"
    assert a.align_sound("دیدم", "didam") == "دِیدَم"


def test_provenance_follows_group_members():
    rs = v.readings_of({"گُل": 3, "گِل": 2}, {"گُل": {"a": 2, "b": 1}, "گِل": {"b": 2}})
    assert rs[0].sources == {"a": 2, "b": 1}
    assert rs[1].sources == {"b": 2}


def test_best_abstains_and_reports_evidence(monkeypatch):
    lex = v.build_lexicon({"گُل": 3, "گِل": 2})
    monkeypatch.setattr(v, "lexicon", lambda: lex)
    assert v.best("گل").form is None
    assert v.best("گل", 0.5).form == "گُل"
    assert v.best("گل").probability == 0.6
    assert v.best("hello").form is None


def test_unknown_model_and_empty_context_abstention(monkeypatch):
    model = {"labels": ["", "َ", "ِ", "ُ", "ْ"], "support": 2,
             "contexts": {"گ": {"ُ": 100}}, "calibration": [0.2] * 10}
    monkeypatch.setattr(v, "lexicon", lambda: v.Lexicon())
    monkeypatch.setattr(v, "_model", lambda: model)
    assert v.best("گل").form is None
    assert v.best("گل", 0.1).form == "گُل"
    assert v.best("کتاب", 0).form is None
    assert v.best("ژ", 0).form is None


def test_nonwords_abstain_before_lookup_or_model(monkeypatch):
    def forbidden():
        raise AssertionError("Invalid words must not access data")
    monkeypatch.setattr(v, "lexicon", forbidden)
    monkeypatch.setattr(v, "_model", forbidden)
    for word in ("", "‌", "‌‌", "آ۱", "آ۔", "۱آ", "گل!", "‌گل", "گل‌", "گ‌‌ل", "َِ"):
        assert v.best(word, 0) == v.Prediction(None, 0.0, False)
        assert v.predict_unknown(word, {})[1] == 0
    assert v.valid_word("می‌روم")
    assert not v.valid_word("می‌ روم")


def test_source_counts_survive_disk_round_trip(tmp_path):
    rs = v.readings_of({"گُل": 3}, {"گُل": {"a": 2, "b": 1}})
    lex = v.Lexicon({"گل": v.Entry("گل", rs, 3, 10, 1)})
    v.write(lex, [], {}, tmp_path)
    assert v.Lexicon.load(tmp_path / "vowels-lexicon.tsv") == lex
    (tmp_path / "old.tsv").write_text("گل\tگُل\t3\t1\t3\t10\t1\n")
    old = v.Lexicon.load(tmp_path / "old.tsv")
    assert old["گل"].readings[0].sources == {}
    assert old["گل"].total == 10


def test_diphthongs():
    assert a.align_sound("بین", "beyn") == "بَیْن"
    assert a.align_sound("پی", "pey") == "پَی"
    assert a.align_sound("مورد", "mored") == "مُورِد"
    assert a.align_sound("مورد", "mored", ow=True) == "مَوْرِد"
    assert a.align_sound("تو", "to", ow=True) == "تُو"
    assert a.align_sound("ویژه", "vi;e", "homorich") == "وِیژِه"
