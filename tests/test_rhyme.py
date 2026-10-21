import pytest

from ghazal_nlp import rhyme
from ghazal_nlp.rhyme_eval import comparable

HAFEZ = """دلم جز مهر مه‌رویان طریقی بر نمی‌گیرد
ز هر در می‌دهم پندش ولیکن در نمی‌گیرد
خدا را ای نصیحتگو حدیث ساغر و می گو
که نقشی در خیال ما از این خوشتر نمی‌گیرد""".split("\n")


def test_a_ghazal_with_a_radif():
    r, = rhyme.scan(HAFEZ, "ghazal")
    assert (r.qafiya, r.radif, r.words, r.letters) == ("ر", "نمی‌گیرد", ("بر", "در", "خوشتر"), "رنمیگیرد")
    assert r.lines == (0, 1, 3) and not r.off and not r.abstention


def test_each_masnavi_couplet_rhymes_on_its_own():
    first, second = rhyme.scan(["بشنو این نی چون شکایت می‌کند", "از جدایی‌ها حکایت می‌کند",
                                "کز نیستان تا مرا ببریده‌اند", "در نفیرم مرد و زن نالیده‌اند"], "masnavi")
    assert (first.qafiya, first.radif) == ("کایت", "می‌کند")
    assert (second.qafiya, second.radif, second.words) == ("یدهاند", "", ("ببریده‌اند", "نالیده‌اند"))


def test_a_robai_rhymes_on_lines_one_two_and_four():
    r, = rhyme.scan(["ای کاش که جای آرمیدن بودی", "یا این ره دور را رسیدن بودی",
                     "کاش از پی صد هزار سال از دل خاک", "چون سبزه امید بردمیدن بودی"], "robai")
    assert (r.qafiya, r.radif, r.lines, r.off) == ("یدن", "بودی", (0, 1, 3), ())


def test_an_opening_line_that_misses_the_rhyme_is_reported():
    r, = rhyme.scan(["سر آن ندارد امشب که برآید آفتابی", "که چه فکرها برآید ز درون خوابی",
                     "تو مگر نه در جهانی که دمی بگیری آرام", "به حجاب اگر نماند برهد ز اضطرابی"], "ghazal")
    assert r.qafiya.endswith("ابی") and r.off == ()
    r, = rhyme.scan(["اول سخن این است", "که چه فکرها برآید ز درون خوابی",
                     "تو مگر نه در جهانی که دمی بگیری آرام", "به حجاب اگر نماند برهد ز اضطرابی"], "ghazal")
    assert r.off == (0,)


def test_the_same_rhyme_word_twice_is_a_fact():
    r, = rhyme.scan(["ما را ز خیال تو چه پروای شراب است", "خم گو سر خود گیر که خمخانه خراب است",
                     "گر خمر بهشت است بریزید که بی دوست", "هر شربت عذبم که دهی عین عذاب است",
                     "افسوس که ایام جوانی همه بگذشت", "این خانه دگر بار ز سیلاب خراب است"], "ghazal")
    assert r.repeated == ("خراب",) and r.radif == "است"


@pytest.mark.parametrize("lines,form,reason", [
    (HAFEZ, "tarji", "form_unknown"),
    (["یک سطر"], "ghazal", "too_few_lines"),
    (["امروز هوا خوب است", "فردا به مدرسه می‌روم", "این جمله قافیه ندارد", "آن یکی هم ندارد"], "ghazal", "no_shared_rhyme"),
])
def test_abstains(lines, form, reason):
    assert rhyme.scan(lines, form)[0].abstention == reason


def test_letters_ignore_how_they_are_encoded():
    assert rhyme.key("خانهٔ") == rhyme.key("خانه‌ی") == "خانهی"
    assert rhyme.key("مَرا،") == "مرا" and rhyme.key("آئین") == "ایین" and rhyme.key("ﮔﻞ") == "گل"
    assert comparable("یدهیاوست") == comparable("یدهاوست")
