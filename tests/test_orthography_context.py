import pytest
from ghazal_nlp import orthography as o


def repairs(rid, text):
    return [f.suggestion for f in o.rulebase().by_id(rid).scan(text)[0]]


@pytest.mark.parametrize("text", ["حمزة را دیدم", "حمزة بن عبدالله", "قناة فلسطين", "هذه قصة طویلة", "جمع عقدة است", "منارة"])
def test_teh_keeps_arabic_names_and_phrases(text):
    assert repairs("ghazal.teh_marbuta", text) == []


@pytest.mark.parametrize("text,expected", [("توسعة اقتصادی", "توسعه‌ی"), ("مقایسة محتوایی", "مقایسه‌ی"),
                                           ("این سورة را بخوان", "سوره")])
def test_teh_preserves_the_ezafe_reading(text, expected):
    assert repairs("ghazal.teh_marbuta", text) == [expected]


@pytest.mark.parametrize("text,expected", [("می\u200fرود", "‌"), ("نوشته\u200eاست", " "),
                                           ("او\u200eاست", " "),
                                           ("نور\u00adها", ""), ("کتاب\u200e‌ها", "")])
def test_control_repairs_follow_the_actual_boundary(text, expected):
    assert repairs("ghazal.zwnj_stand_in", text) == [expected]


@pytest.mark.parametrize("text", ["\u200fسلام", "C++\u200e، زبان", "نام \u200eمتن", "سان\u200eفرانسیسکو", "دکتر\u200eحسن"])
def test_controls_can_be_intentional_or_uncertain(text):
    assert repairs("ghazal.invisible_controls", text) == []
    assert repairs("ghazal.zwnj_stand_in", text) == []


def test_only_proven_silent_he_requires_plural_separation():
    rid = "academy.plural_ha_after_silent_he"
    for word in ("گروهها", "دهها", "کوهها", "سروگروههای"):
        assert repairs(rid, word) == []
    assert repairs(rid, "خانهها فرضيهها") == ["خانه‌ها", "فرضیه‌ها"]


def test_clitic_names_and_plural_linker():
    rid = "academy.clitic_pronoun_after_vowel_host"
    assert repairs(rid, "تیموتی اش") == []
    assert repairs(rid, "دست‌های تان دست‌های اش") == ["دست‌هایتان", "دست‌هایش"]
    assert repairs(rid, "شخصی اش") == ["شخصی‌اش"]


def test_mi_normalizes_letters_but_does_not_guess_unattested_solid_forms(monkeypatch):
    assert repairs("ghazal.mi_prefix", "می كنند") == ["می‌کنند"]
    assert repairs("ghazal.mi_prefix", "میلان میدان میوه") == []
    monkeypatch.setattr(o, "closed_list", lambda _: {})
    detector = o.VerbPrefix("test", "orthography", solid_file="empty")
    assert list(detector.scan("میرود", o.tokenize("میرود"))) == []
    assert [f.suggestion for f in detector.scan("می رود", o.tokenize("می رود"))] == ["می‌رود"]


@pytest.mark.parametrize("text", ["`سلام،خوبی`", "<code>سلام،خوبی</code>", "```سلام،خوبی```",
                                  "https://x/سلام،خوبی", "<a title='سلام،خوبی'>"])
def test_punctuation_keeps_code_and_links(text):
    assert repairs("ghazal.space_after_punctuation", text) == []
    assert repairs("ghazal.latin_punctuation", text.replace("،", ",")) == []
    assert repairs("ghazal.space_after_punctuation", "سلام،خوبی") == ["، "]


def test_latin_punctuation_includes_spacing_and_keeps_latin_context():
    assert repairs("ghazal.latin_punctuation", "کتاب ,قلم") == ["، "]
    assert repairs("ghazal.latin_punctuation", "سلام ;-)") == []
    assert repairs("ghazal.latin_punctuation", 'Author, "A long English title", عنوان, 2020') == []


@pytest.mark.parametrize("text,expected", [("شماره‌12", "شماره "), ("رشته‌{field}", "رشته "),
                                           ("نوشتیم‌،", "نوشتیم")])
def test_redundant_zwnj_may_need_a_real_space(text, expected):
    assert repairs("ghazal.redundant_zwnj", text) == [expected]
    assert repairs("ghazal.redundant_zwnj", "صفت‌\u200eهای خوب") == []


def test_quotes_are_complete_bounded_persian_pairs():
    rid = "ghazal.guillemets"
    for text in ('"O" و "H"', 'مثلا" این را مجددا" بنویس', '"العطاء لا يكون عن خوف"', '"'+'الف '*100+'"'):
        assert repairs(rid, text) == []
    assert repairs(rid, '" سلام "گفت') == ["«سلام» "]


def test_conjunction_is_distinct_from_bana_bar_this_noun():
    rid = "ghazal.joined_always"
    for text in ("بنا بر این گزارش", "بنا بر این تفسیر", "بنا بر این بود که", "بنا بر این شیوه"):
        assert repairs(rid, text) == []
    assert repairs(rid, "بنا بر این رفتیم") == ["بنابراین"]
    assert repairs(rid, "بنا بر این. گزارش رسید.") == ["بنابراین"]


def test_ezafe_hamza_abstains_on_unproven_hosts():
    assert repairs("ghazal.ezafe_hamza_letter", "خانهء من") == ["خانه‌ی"]
    assert repairs("ghazal.ezafe_hamza_letter", "قهقهء کوهء") == []


def test_content_splitter_stays_off_even_when_its_table_matches():
    rid = "ghazal.content_splits"
    rule = o.rulebase().by_id(rid)
    assert rule.status == "retired" and not rule.runs
    assert not [f for f in o.rulebase().check("آخرینلحظه کماست وایبر", min_confidence="low").findings if f.rule_id == rid]


def test_context_tables_have_per_file_provenance():
    licence = (o.DATA/"LICENSE").read_text()
    for name in ("mi-forms", "clitic-hosts", "context-names"):
        path = o.DATA/("orthography-"+name+".tsv")
        header = '\n'.join(l for l in path.read_text().splitlines() if l.startswith('#'))
        assert path.name in licence and "Licence: CC BY-SA 4.0" in header and "held-out documents excluded" in header or "training rows" in header
        sources = ("ud",) if name == "context-names" else ("fawiki", "mizan", "irib-news", "blogfa", "books", "persian-tweets-2024", "ganjoor")
        if name == "context-names":
            assert "Seraji" in header and "PerDT" in header and "training" in header
        else:
            assert all(source in header for source in sources)
