import tomllib

import pytest

from ghazal_nlp.orthography import (
    RULES, ZWNJ, Abstention, AffixZwnj, ClosedList, RuleBase, RuleError, closed_list, is_silent_he,
    parse_rule, rulebase, silent_he_inventory, tokenize,
)

BOOK = "دستور خطّ فارسی"
SHA = "6071a6fe7572dfb6b8211033bba82436d59bbe8ec3c1f81312de6b76ef90603e"
CITED = [
    "orthography.demonstrative_compounds", "orthography.har_lexicalised", "orthography.hamin_haman_lexicalised",
    "orthography.hich_lexicalised", "orthography.ke_compound", "orthography.compound_conjunction",
    "orthography.compound_preposition", "orthography.numeral_spacing", "orthography.copula_after_silent_he",
    "orthography.clitic_pronoun_after_silent_he", "orthography.ya_nakare_after_silent_he",
    "orthography.hamze_medial_carrier", "orthography.hamze_final_carrier", "orthography.hamze_not_yeh",
    "orthography.taa_marbuta_word_final", "orthography.tanvin_nasb_mandatory", "orthography.double_vav_syllable",
    "orthography.preferred_spelling",
]


def record(**overrides):
    base = {
        "id": "orthography.t", "layer": "orthography", "title": {"fa": "x"}, "statement": {"fa": "x"},
        "kind": "prescriptive", "status": "active", "scope": {"registers": ["formal_written"]},
        "source": {"work": "w", "page": 1}, "examples": {"ok": ["الف"], "violations": [{"text": "ب"}]},
    }
    return base | overrides


def scan(rule_id, text):
    return rulebase().by_id(rule_id).scan(text)


def found(rule_id, text):
    return scan(rule_id, text)[0]


def suggestions(rule_id, text):
    return [f.suggestion for f in found(rule_id, text)]


# --- the invariants -----------------------------------------------------------


def test_the_packs_load_verified_and_count_what_runs():
    base = RuleBase.load(verify=True)
    s = base.summary()
    assert (s["recorded"], s["executable"], s["running"]) == (45, 34, 33)
    assert s["by_status"] == {"proposed": 12, "active": 31, "contested": 2}


def test_every_rule_cites_a_page_and_declares_its_registers():
    for rule in rulebase().rules:
        assert rule.source.work and rule.source.page and "ص" in rule.source.citation(), rule.id
        assert rule.registers, rule.id
        if BOOK in rule.source.work:
            assert rule.source.isbn == "978-622-5305-22-9" and rule.source.sha256 in (None, SHA), rule.id


@pytest.mark.parametrize("rule_id", CITED)
def test_the_page_is_inside_the_book(rule_id):
    rule = rulebase().by_id(rule_id)
    pages = [int(p) for p in rule.source.page.replace("–", " ").replace(",", " ").split()]
    assert pages and all(1 <= p <= 86 for p in pages) and rule.source.sha256


def test_pack_ids_are_unique():
    ids = [r["id"] for p in RULES.glob("*.toml") for r in tomllib.loads(p.read_text(encoding="utf-8"))["rule"]]
    assert len(ids) == len(set(ids)) == 45


@pytest.mark.parametrize("overrides,message", [
    ({"source": {"work": "w"}}, "source.page"),
    ({"scope": {"registers": []}}, "registers"),
    ({"scope": {"registers": ["slang"]}}, "unknown registers"),
    ({"examples": {"ok": ["الف"]}}, "both examples"),
    ({"title": {"en": "x"}}, "title.fa"),
    ({"detector": {"type": "oracle"}}, "unknown detector"),
])
def test_a_malformed_record_is_rejected(overrides, message):
    with pytest.raises(RuleError, match=message):
        parse_rule(record(**overrides))


def test_a_rule_that_fails_its_examples_raises():
    rule = parse_rule(record(detector={"type": "affix_zwnj", "affix": "تر"},
                             examples={"ok": ["بزرگتر"], "violations": [{"text": "بزرگ تر"}]}))
    with pytest.raises(RuleError, match="was flagged"):
        rule.verify()
    rule = parse_rule(record(detector={"type": "affix_zwnj", "affix": "تر"},
                             examples={"ok": ["بزرگ‌تر"], "violations": [{"text": "بزرگ تر", "expect": "بزرگتر"}]}))
    with pytest.raises(RuleError, match="expected repair"):
        rule.verify()


def test_a_pack_that_fails_does_not_load(tmp_path):
    (tmp_path / "bad.toml").write_text(
        '[[rule]]\nid = "orthography.t"\nlayer = "orthography"\nkind = "prescriptive"\nstatus = "active"\n'
        'title.fa = "x"\nstatement.fa = "x"\nscope.registers = ["formal_written"]\n'
        'source.work = "w"\nsource.page = 1\ndetector.type = "affix_zwnj"\ndetector.affix = "تر"\n'
        'examples.ok = ["بزرگتر"]\nexamples.violations = ["بزرگ تر"]\n', encoding="utf-8")
    with pytest.raises(RuleError):
        RuleBase.load([tmp_path / "bad.toml"])
    with pytest.raises(RuleError, match="duplicate"):
        RuleBase.load([RULES / "orthography-academy.toml"] * 2, verify=False)


def test_recorded_rules_may_have_no_detector():
    recorded = [r for r in rulebase().rules if not r.executable]
    assert recorded and all(r.examples_ok and r.examples_bad for r in recorded)


# --- the engine ---------------------------------------------------------------


def test_tokens_keep_zwnj_and_findings_point_into_the_text():
    assert [t.text for t in tokenize(f"بزرگ{ZWNJ}تر است")] == [f"بزرگ{ZWNJ}تر", "است"]
    text = f"کتاب ها را می{ZWNJ}خوانم"
    analysis = rulebase().check(text)
    assert analysis.findings and analysis.rulebase_version == "0.1.0"
    assert all(text[slice(*f.span)] == f.text and BOOK in f.citation for f in analysis.findings)


def test_register_scope_is_enforced_at_run_time():
    assert rulebase().check("کتاب ها").findings
    assert not any(f.rule_id == "orthography.plural_ha"
                   for f in rulebase().check("کتاب ها", registers=("colloquial",)).findings)


def test_contested_rules_say_so():
    contested = [f for f in rulebase().check(f"خانه{ZWNJ}ی او").findings if f.contested]
    assert contested[0].rule_id == "orthography.ezafe_after_silent_he"


def test_confidence_floor_drops_weak_findings():
    assert rulebase().check("بزرگتر", min_confidence="medium").findings
    assert not rulebase().check("بزرگتر", min_confidence="high").findings


def test_profile_measures_coverage_and_abstention():
    report = rulebase().profile(["کتاب ها", f"کتاب{ZWNJ}ها"])
    assert (report["documents"], report["documents_with_finding"]) == (2, 1)
    assert "abstentions_by_reason" in report


def test_clean_prose_produces_no_finding():
    text = ("هیچ‌کس نمی‌داند که این کتاب را چه کسی نوشته است. "
            "نویسنده‌اش سال‌ها پیش درگذشته و نسخه‌های خطی آن در کتابخانه‌های "
            "بزرگ‌تر نگهداری می‌شود. هرکس بخواهد آن را بخواند باید اجازه بگیرد.")
    assert rulebase().check(text).findings == ()


def test_unruled_compounds_stay_unruled():
    for text in ("نخست وزیر ایران", "نخست‌وزیر ایران"):
        assert not rulebase().check(text).findings


# --- detectors ----------------------------------------------------------------


def test_a_short_host_abstains():
    d = AffixZwnj(rule_id="t", layer="orthography", affix="تر")
    assert [r.reason for r in d.scan("دکتر", tokenize("دکتر"))] == ["host_too_short_to_decide"]


def test_ham_prefix_fires_only_on_its_lexicon():
    assert any(f.rule_id == "orthography.ham_prefix" for f in rulebase().check("هم وطن").findings)
    assert not any(f.rule_id == "orthography.ham_prefix" for f in rulebase().check("با هم آمدند").findings)


def closed(text, **kwargs):
    return list(ClosedList(rule_id="t", layer="orthography", **kwargs).scan(text, tokenize(text)))


def test_closed_list_matches_whole_tokens_and_phrases():
    (f,) = closed("اطاق کوچک", entries={"اطاق": "اتاق"})
    assert (f.suggestion, f.span) == ("اتاق", (0, 4))
    assert closed("لقمه", entries={"لق": "لق"}) == closed("اطاقی", entries={"اطاق": "اتاق"}) == []
    assert len(closed("هر کس", entries={"هر کس": "هرکس"})) == 1
    assert closed("هر\nکس", entries={"هر کس": "هرکس"}) == closed("هر، کس", entries={"هر کس": "هرکس"}) == []
    assert [f.text for f in closed("این واقعا، درست است", entries={"واقعا": "واقعاً"})] == ["واقعا"]


def test_closed_list_abstains_by_name():
    (a,) = closed("هر چند سال یک بار", entries={"هر کس": "هرکس"}, abstain_on=frozenset({"هر چند"}),
                  abstain_reason="ambiguous")
    assert isinstance(a, Abstention) and a.reason == "ambiguous"


@pytest.mark.parametrize("name,absent", [
    ("orthography-preferred-spellings.tsv", ("آخر", "تراز", "طراز", "طاق", "تاق", "طاس", "تاس", "غرق", "حیز", "عنتر")),
    ("orthography-tanvin-nasb.tsv", ("ابدا", "اصلا", "مطلقا", "جدا", "بنا", "عرفا")),
])
def test_closed_list_files_hold_no_sense_dependent_row(name, absent):
    table = closed_list(name)
    assert table and all(w and r and w != r for w, r in table.items())
    assert not set(absent) & set(table)


# --- the rules ------------------------------------------------------------------


def test_copula_leaves_the_vocative_alone():
    assert found("orthography.copula_after_silent_he", "به خانه ای دوست") == []
    assert found("orthography.copula_after_silent_he", "خسته است") == []


def test_taa_marbuta_leaves_compounds_and_titles_and_offers_no_repair():
    rule = "orthography.taa_marbuta_word_final"
    assert found(rule, "دایرةالمعارف بزرگ") == found(rule, "تذکرة الاولیاء") == []
    assert [f.text for f in found(rule, "رحمة")] == ["رحمة"]
    assert suggestions(rule, "معاویة") == [None]


def test_lexicalised_lists_abstain_on_their_ambiguous_members():
    got, abstained = scan("orthography.har_lexicalised", "هر چند سال یک بار")
    assert got == [] and [a.text for a in abstained] == ["هر چند"]
    assert found("orthography.demonstrative_compounds", "این چه کاری است") == []
    got, abstained = scan("orthography.demonstrative_compounds", "این که حرف تازه‌ای نیست")
    assert got == [] and [a.text for a in abstained] == ["این که"]
    assert found("orthography.hich_lexicalised", "هیچ چیز") == []
    assert suggestions("orthography.hich_lexicalised", "هیچ کس") == [f"هیچ{ZWNJ}کس"]


@pytest.mark.parametrize("rule,text,reason", [
    ("orthography.ya_nakare_after_silent_he", "پندم مده ای دوست", "he_not_proven_silent"),
    ("orthography.ya_nakare_after_silent_he", "نکردی جگرگاه ای پور باز", "he_is_voiced"),
    ("orthography.ya_nakare_after_silent_he", "بگفتا که ای یار", "host_too_short_or_not_persian"),
    ("orthography.copula_after_silent_he", "از آن آگاه اند", "he_is_voiced"),
])
def test_the_silent_he_gate_declines_by_name(rule, text, reason):
    got, abstained = scan(rule, text)
    assert got == [] and [a.reason for a in abstained] == [reason]


def test_enclitics_join_on_a_proven_or_vowel_host():
    assert suggestions("orthography.ya_nakare_after_silent_he", "در گوشه ای خزیدم") == [f"گوشه{ZWNJ}ای"]
    assert suggestions("orthography.clitic_pronoun_after_vowel_host", "زندگی اش") == [f"زندگی{ZWNJ}اش"]
    assert suggestions("orthography.copula_after_vowel_host", "ایرانی ام") == [f"ایرانی{ZWNJ}ام"]


def test_the_indefinite_after_a_ya_host_is_recorded_and_not_run():
    rule = rulebase().by_id("orthography.ya_nakare_after_vowel_host")
    assert rule.status == "proposed" and rule.executable and not rule.runs
    assert found("orthography.ya_nakare_after_silent_he", "شکی ای دوست") == []


def test_compounds_after_a_silent_he():
    rule = "orthography.compound_after_silent_he"
    assert set(suggestions(rule, "او علاقه مند بود و عشوه گری می‌کرد")) == {"علاقه‌مند", "عشوه‌گری"}
    for text in ("خانه من بزرگ است", "خانه او را دیدم", "نامه دوست رسید", "همه کار او", "جمله شناس", "همیشه بان"):
        assert found(rule, text) == [], text
    got, abstained = scan(rule, "گره مند")
    assert got == [] and abstained and all(isinstance(a, Abstention) for a in abstained)


def test_the_separator_follows_the_hosts_last_letter():
    assert suggestions("orthography.suffix_always_zwnj", "سمندر وش") == ["سمندروش"]
    assert suggestions("orthography.suffix_always_zwnj", "ارغوان وار") == ["ارغوان‌وار"]


def test_homographs_left_out_of_the_lists():
    for text in ("به نام خدا", "به دین اسلام گروید", "به روز بعد موکول شد"):
        assert found("orthography.compound_lexicalised_solid", text) == [], text
    assert found("orthography.arabic_term_phrase", "من بعد ذلک غفراناً لکاتبه") == []


# --- the silent ه -------------------------------------------------------------


@pytest.mark.parametrize("word,expected", [
    ("خانه", True), ("میوه", True), ("سایه", True), ("قهوه", True), ("گوشه", True),
    ("راه", False), ("ماه", False), ("گاه", False), ("جگرگاه", False), ("سیاه", False),
    ("گره", None), ("فقیه", None), ("مده", None), ("که", None), ("کتاب", None),
])
def test_is_silent_he_answers_or_abstains(word, expected):
    assert is_silent_he(word) is expected


def test_the_inventory_holds_only_what_it_could_prove():
    inventory = silent_he_inventory()
    assert len(inventory) == 1269
    assert all(w.endswith("ه") and len(w) >= 3 and not w.endswith("اه") for w in inventory)
    assert is_silent_he(f"فیلم{ZWNJ}نامه") is is_silent_he("فیلمنامه")
