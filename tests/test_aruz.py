from dataclasses import replace
from itertools import product

import pytest

from ghazal_nlp import aruz, aruz_eval
from ghazal_nlp.aruz import (
    DATA, Devices, Inventory, Marks, MetreEntry, Scanner, Scansion, WeightEntry, WeightLexicon,
    dowri_half, pattern_for_label, prepare, scan_poem,
)
from ghazal_nlp.aruz_eval import classify, readings, realizes, weights_of

HAZAJ = "مفاعیلن مفاعیلن مفاعیلن مفاعیلن (هزج مثمن سالم)"
DOBEYTI = "مفاعیلن مفاعیلن فعولن (هزج مسدس محذوف)"
SHAHNAMEH = "فعولن فعولن فعولن فعل (متقارب مثمن محذوف)"
RAMAL = "فاعلاتن فاعلاتن فاعلاتن فاعلن (رمل مثمن محذوف)"
RAJAZ = "مفتعلن مفاعلن مفتعلن مفاعلن (رجز مثمن مطوی مخبون)"
FIRST_PUBLISHED = Devices(ezafe_long=False, tashdid_before_vowel=False, ast_contraction=False)
EL = Devices(tashdid_before_vowel=False)
EVERY_PATTERN = ["".join(p) for n in range(1, 9) for p in product("u-", repeat=n)]


def inventory(*labels, poems=0):
    return Inventory(MetreEntry(label, pattern_for_label(label), "arkan", poems) for label in labels)


@pytest.fixture(scope="module")
def small():
    return inventory(HAZAJ, DOBEYTI, SHAHNAMEH, RAMAL)


@pytest.fixture(scope="module")
def full():
    return Inventory.load()


def read(line, marks=True, devices=EL):
    return readings(prepare(line, marks=Marks() if marks else None), EVERY_PATTERN, devices)


def r(tokens):
    return tuple(tokens.split())


def letter_readings(word):
    skel = prepare(word)
    reach = [set() for _ in range(len(skel) + 1)]
    reach[0].add("")
    for i in range(len(skel)):
        for weights, nxt, _ in aruz._syllables(skel, i, Devices()):
            reach[nxt] |= {prefix + weights for prefix in reach[i]}
    return reach[len(skel)]


def test_feet_expand_to_their_pronunciation():
    assert pattern_for_label("فاعلاتن") == "-u--"
    assert pattern_for_label("مفاعیلن مفاعیلن فعولن") == "u---u---u--"
    assert pattern_for_label("فاعلاتن فاعلاتن فاعلن (رمل مسدس محذوف یا وزن مثنوی)") == "-u---u---u-"
    assert pattern_for_label("فاعلاتن نامعلوم") is None


def test_preparation():
    skel = prepare("يك كتاب")
    assert skel.words == ("یک", "کتاب") and "".join(skel.chars) == "یککتاب"
    assert prepare("می‌رود").words == ("میرود",)
    skel = prepare("دل بر")
    assert (skel.word, skel.initial, skel.final) == ((0, 0, 1, 1), (True, False, True, False), (False, True, False, True))


def test_abstentions_are_named(small):
    scan = Scanner(inventory=small).scan
    assert scan("hello 123").abstention == "non_persian_script"
    assert scan("   ").abstention == "empty_line"
    assert scan("دل").abstention == "length_outside_inventory"


def test_known_hemistichs(small):
    s = Scanner(inventory=small).scan("الا یا ایها الساقی ادر کأسا و ناولها")
    assert s.metres == (HAZAJ,) and s.weights[0] == "u---u---u---u---"
    assert set(Scanner(inventory=small).scan("بیا ساقی آن می که حال آورد").metres) == {DOBEYTI, SHAHNAMEH}
    s = Scanner(inventory=small).scan("مرا با توست چندین آشنایی")
    assert s.metres and s.devices and set(s.devices) <= set(aruz.DEVICES)


def test_the_shahnameh_needs_the_conjunction_vav():
    inv = inventory(SHAHNAMEH)
    s = Scanner(inventory=inv).scan("خداوند نام و خداوند جای")
    assert s.metres == (SHAHNAMEH,) and "conjunction_vav" in s.devices
    off = Devices(conjunction_vav=False, ezafe_long=False, tashdid_before_vowel=False)
    assert not Scanner(inventory=inv, devices=off).scan("خداوند نام و خداوند جای").metres


def test_costs_rank_readings(small):
    s = Scanner(inventory=small).scan("الا ای آهوی وحشی کجایی")
    costs = dict(s.candidates)
    assert len(costs) > len(s.metres) and s.licences == min(costs[m] for m in s.metres)
    assert costs[DOBEYTI] == 0 < costs[SHAHNAMEH] and s.metres == (DOBEYTI,)
    assert aruz.COST[aruz.FINAL] == 0 and aruz.COST[aruz.EBDAL] == 1


def test_a_device_off_abstains_rather_than_guesses(small):
    strict = Scanner(inventory=small, devices=Devices(
        hamze_elision=False, conjunction_vav=False, ezafe_glide=False, silent_vav_after_khe=False,
        overlong_as_long=False, ebdal=False, initial_long=False, tashdid=False))
    lines = ["الا یا ایها الساقی ادر کأسا و ناولها", "به نام خداوند جان و خرد", "مرا با توست چندین آشنایی"]
    assert any(not strict.scan(line).metres for line in lines)


def test_the_inventory(full):
    assert len(full) == 65 and len(Inventory.load(min_poems=1)) == 139
    for e in full.entries:
        assert set(e.pattern) <= {"u", "-"} and e.provenance in {"arkan", "derived", "unexpanded"}
        if e.provenance == "arkan":
            assert e.pattern == pattern_for_label(e.label)
    rubai = [e for e in full.entries if aruz.RUBAI in e.label]
    assert len(rubai) > 1 and any(e.provenance == "derived" for e in rubai)


def test_poems(full):
    assert aruz.hemistichs("عنوان\nالا ای آهوی وحشی کجایی\n\nدل\n") == ["الا ای آهوی وحشی کجایی"]
    poem = "\n".join(["الا یا ایها الساقی ادر کأسا و ناولها", "که عشق آسان نمود اول ولی افتاد مشکل ها",
                      "به بوی نافه ای کآخر صبا زان طره بگشاید", "ز تاب جعد مشکینش چه خون افتاد در دل ها"])
    result = scan_poem(poem, Scanner(inventory=full))
    assert HAZAJ in result.metres and result.support == result.lines == 4
    result = scan_poem("الا ای آهوی وحشی کجایی\nمرا با توست چندین آشنایی\nاین خط ۱۲۳ عدد دارد", Scanner(inventory=full))
    assert (result.lines, result.abstained, result.abstentions) == (2, 1, (("non_persian_script", 1),))
    assert not scan_poem("۱۲۳ این خط عدد دارد و شعر نیست", Scanner(inventory=full)).metres


def test_the_split_is_a_stable_fifth():
    held = [n for n in range(5000) if aruz_eval.held_out(str(n))]
    assert 0.17 < len(held) / 5000 < 0.23


def test_elision_and_the_conjunction_suspend_the_lexicon():
    assert aruz._constrainable_words(prepare("در آن باغ"), Devices()) == [False, False, True]
    assert aruz._constrainable_words(prepare("جان و خرد"), Devices())[:2] == [False, False]


def test_a_pinned_word_prunes_its_other_cuts():
    assert aruz._word_cuts("دل", frozenset({"-"}), Devices()) == frozenset({(0, "-", 2)})
    assert "uu" in letter_readings("دل")
    assert aruz._word_cuts("دل", frozenset({"---"}), Devices()) is None


def test_the_gate(small):
    lexicon = WeightLexicon([WeightEntry("دل", aruz.PINNED, ("-",), 60, 1.0, (("-", 60),))])
    gate = Scanner(inventory=small, lexicon=lexicon)._gate(prepare("دل من"))
    assert gate[0] == frozenset({("-", 2)}) and gate[1] == frozenset() and gate[2] is gate[3] is None
    assert Scanner(inventory=small)._gate(prepare("دل من")) is None


def test_a_lexicon_that_cannot_fit_backs_off(small):
    line = "الا ای آهوی وحشی کجایی"
    plain = Scanner(inventory=small).scan(line)
    assert "u-uuu" in letter_readings("کجایی")
    wrong = WeightLexicon([WeightEntry("کجایی", aruz.PINNED, ("u-uuu",), 30, 1.0, (("u-uuu", 30),))])
    gated = Scanner(inventory=small, lexicon=wrong)
    assert not gated._search(prepare(line), gated._gate(prepare(line)))
    assert gated.scan(line).metres == plain.metres and not gated.scan(line).gated


def test_what_the_evidence_says_about_a_word():
    assert classify({frozenset({"-", "uu"}): 50, frozenset({"-"}): 50}, 4, 1.0)[:3] == (aruz.PINNED, ("-",), 100)
    dissent = {frozenset({"-"}): 199, frozenset({"u"}): 1}
    assert classify(dissent, 4, 0.99)[:2] == (aruz.PINNED, ("-",))
    assert classify(dissent, 4, 1.0)[0] == aruz.VARIABLE
    assert classify({frozenset({"-", "u"}): 5}, 4, 1.0)[:2] == (aruz.UNDERDETERMINED, ("-", "u"))
    assert classify({frozenset({"u"}): 60, frozenset({"-"}): 40}, 4, 1.0)[:2] == (aruz.VARIABLE, ("-", "u"))
    status, weights, _, support = classify({frozenset({w}): 10 for w in ("-", "u", "--", "uu", "-u")}, 4, 0.99)
    assert (status, weights, len(support)) == (aruz.CONFLICT, (), 5)
    assert not WeightLexicon([WeightEntry("گل", status, weights, 50, 0.0, tuple(support))]).allowed
    once = WeightEntry("دلبر", aruz.PINNED, ("--",), 1, 1.0, (("--", 1),))
    assert not WeightLexicon([once]).allowed and WeightLexicon([once], min_occurrences=1).allowed


def test_the_lexicon_round_trips(tmp_path):
    entries = [
        WeightEntry("دل", aruz.PINNED, ("-",), 64199, 0.9993, (("-", 64154), ("uu", 19871))),
        WeightEntry("به", aruz.VARIABLE, ("-", "u"), 198882, 1.0, (("u", 193137),)),
        WeightEntry("عشق", aruz.CONFLICT, (), 254, 0.0, (("-", 207), ("-u", 206))),
    ]
    aruz_eval.write_weights(entries, {"poems": 99591}, tmp_path / "w.tsv")
    loaded = WeightLexicon.load(tmp_path / "w.tsv")
    assert list(loaded.entries) == entries and loaded.provenance["poems"] == "99591"
    assert "supervised" in loaded.provenance["supervision"] and "development only" in loaded.provenance["split"]


def test_the_shipped_lexicon_keeps_the_plain_reading():
    allowed = WeightLexicon.load().allowed
    assert "--" in allowed["یارم"] and "--" in allowed["دوری"]


def test_votes_and_winners(monkeypatch):
    rubai = "مفعول مفاعیل مفاعیل فعل (وزن رباعی)"
    dobeyti = "مفاعیلن مفاعیلن فعولن (هزج مسدس محذوف یا وزن دوبیتی)"
    costs = {rubai: 3, dobeyti: 9}
    assert aruz.winners({rubai: 10, dobeyti: 10}, costs, 10) == [dobeyti]
    assert aruz.winners({rubai: 4, dobeyti: 4}, costs, 4) == [rubai]
    assert aruz.winners({rubai: 10, dobeyti: 9}, costs, 10) == [rubai]
    line = Scansion("x", metres=("a",), licences=1, candidates=(("a", 1), ("b", 2), ("c", 9)))
    monkeypatch.setattr(aruz, "VOTE_MARGIN", 1)
    assert aruz.votes(line) == [("a", 1), ("b", 2)]


def test_vowel_licences_are_already_in_the_letters():
    assert {"-u-"} <= letter_readings("نگذرد") and "--" in letter_readings("نبود")
    assert "-u" in letter_readings("دلش") and "u-" in letter_readings("سوی")


def test_one_metre_one_label():
    major, minor = "فعلاتن مفاعلن فعلن (خفیف مسدس مخبون)", "فاعلاتن مفاعلن فعلن"
    inv = Inventory([MetreEntry(major, pattern_for_label(major), "arkan", 5836),
                     MetreEntry(minor, pattern_for_label(minor), "arkan", 29)])
    assert inv.pattern_labels[pattern_for_label(minor)] == [major]
    s = Scanner(inventory=inv).scan("هر که در بند زلف یار بود")
    assert s.patterns == (pattern_for_label(minor),) and s.metres == (major,)
    named, bare = "فاعلاتن فاعلاتن فاعلن (رمل مسدس محذوف یا وزن مثنوی)", "فاعلاتن فاعلاتن فاعلن"
    p = pattern_for_label(bare)
    inv = Inventory([MetreEntry(named, p, "arkan", 5501), MetreEntry(bare, p, "arkan", 2)])
    assert inv.pattern_labels[p] == [named] and len(inv) == 2


def test_the_dowri_caesura():
    assert dowri_half(RAJAZ) == 8
    assert dowri_half("مستفعلن فع مستفعلن فع (متقارب مثمن اثلم)") == 5
    for label in (HAZAJ, "مستفعلتن مستفعلتن", "مفعول مفاعیل مفاعیل فعل (وزن رباعی)",
                  "مفاعلن فعلاتن مفاعلن فعلاتن (مجتث مثمن مخبون)"):
        assert dowri_half(label) is None
    dowri = Scanner(inventory=inventory(RAJAZ))
    assert "overlong_as_long" in dowri.scan("از همه کس کناره‌گیر صحبت آشنا طلب").devices
    line = "ای که ز دیده رفته ناگهان دل ما شکسته‌ای"
    assert not dowri.scan(line).metres
    assert Scanner(inventory=inventory(RAJAZ), devices=Devices(caesura=False)).scan(line).metres
    mashkul = inventory("فعلات فاعلاتن فعلات فاعلاتن (رمل مثمن مشکول)")
    assert Scanner(inventory=mashkul).scan("تن آدمی شریف است بجان آدمیت").licences == 2


def test_marks_sit_beside_the_letters():
    assert prepare("مَدِه جامِ") == prepare("مده جام") and prepare("مَدِه جامِ").marks == ()
    marked = prepare("مَدِه جامِ", marks=Marks())
    assert marked.chars == prepare("مده جام").chars and marked.marks == (aruz.FATHA, aruz.KASRA, "", "", "", aruz.KASRA)
    marked = prepare("حلقۀ", marks=Marks())
    assert "".join(marked.chars) == "حلقه" and marked.marks[-1] == aruz.HAMZA
    marked = prepare("خانه‌ها", marks=Marks())
    assert marked.words == ("خانهها",) and marked.final[3] and marked.initial[4] and not prepare("خانه‌ها").final[3]
    for line in ("دل از کفر و دست از خیانت نشست", "خدا سرمایه امید و بیم است"):
        assert Scanner(devices=EL).scan(line) == Scanner(devices=EL, marks=None).scan(line)


def test_each_mark_family():
    assert r("S -") in read("دشتِ من", marks=False) and r("S -") not in read("دشتِ من") and r("- u -") in read("دشتِ من")
    assert r("- - -") in read("دشتِ من") and r("- - -") not in read("دشتِ من", devices=FIRST_PUBLISHED)
    assert r("- u -") in read("جان من", marks=False) and r("- u -") not in read("جانْ من")
    got = read("خانهٔ من")
    assert r("- u -") not in got and {r("- u u -"), r("- u - -"), r("- - -")} <= got
    assert r("- u -") in read("خانه من", marks=False)
    got = read("راهِ من")
    assert r("- u -") in got and r("S -") not in got and r("S -") in read("راه من", marks=False)
    assert {r("- u u -"), r("- - u -")} <= read("شاه یمن", marks=False)
    assert r("- -") not in read("حقا", marks=False) and read("حقّا") == {r("- -")}
    assert r("- u -") in read("دُرِّ من") and r("u u -") not in read("دُرِّ من")
    assert r("- u S") in read("پرّ زاغ") and r("- u S") not in read("پر زاغ", marks=False) and r("-") in read("حقّ")
    assert r("- u -") in read("دیده‌ام") and r("- u -") not in read("دیده‌ام", marks=False)
    assert r("- u -") in read("جان وُ دل") and r("- u -") not in read("جان وَ دل")


def test_licences():
    assert r("u S") in read("بخوان", marks=False, devices=Devices())
    assert r("u S") not in read("بخوان", marks=False, devices=Devices(silent_vav_after_khe=False))
    for line, reading in (("نبشته است", "u - S"), ("حلاوتی است", "u - u S"), ("پنهان ست", "- S")):
        assert r(reading) in read(line, marks=False, devices=Devices())
        assert r(reading) not in read(line, marks=False, devices=Devices(ast_contraction=False))
    skel = prepare("من من من")
    barred = Devices(ebdal_first_foot=False)
    assert r("- - -") in readings(skel, ["uu--"], Devices())
    assert r("- - -") not in readings(skel, ["uu--"], barred) and r("- - -") in readings(skel, ["--uu"], barred)


def test_the_reading_notation():
    assert weights_of("xâ.ne.ye man") == r("- u u -") and weights_of("dast") == r("S")
    assert weights_of("'aq.lē") == r("- -") and weights_of("pa.zi.râ.yĭ") == r("u - - u")
    assert weights_of("mey mowj") == r("- S")
    with pytest.raises(ValueError):
        weights_of("xt")
    assert realizes(weights_of("ma.nam tang.del tâ šo.dam tang.dast"), "u--u--u--u-")
    assert not realizes(weights_of("ma.nam tang.del tâ šo.dam"), "u--u--u--u-")


@pytest.mark.parametrize("name", ["readings-thiesen.tsv"])
def test_every_reading_realises_its_metre(full, name):
    rows = aruz_eval.load_readings(DATA / name)
    assert len(rows) in (513, 34)
    for row in rows:
        assert row["provenance"].startswith("thiesen-1982:p."), row["id"]
        assert any(realizes(weights_of(row["pronunciation"]), p) for p in full.label_patterns[row["metre"]]), row["id"]


def test_the_scanner_reaches_the_readings():
    rows = aruz_eval.load_readings(DATA / "readings-thiesen.tsv")
    assert aruz_eval.score_readings(rows, Scanner())["reading_admitted"] == 494
