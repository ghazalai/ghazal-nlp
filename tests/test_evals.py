import json

from ghazal_nlp import aruz, aruz_baselines, g2p_eval, orthography, orthography_eval, verb, verb_eval
from ghazal_nlp.aruz_eval import held_out


def test_orthography_eval(tmp_path):
    wrong, correct = tmp_path / "wrong.txt", tmp_path / "correct.txt"
    wrong.write_text("کتاب ها را میخوانند\nکتاب ها را میخوانند\nاو اثر گذار است\n", encoding="utf-8")
    correct.write_text("کتاب‌ها را می‌خوانند\nکتاب‌ها را می‌خوانند\nاو اثرگذار است\n", encoding="utf-8")
    r = orthography_eval.evaluate([(wrong, correct)], orthography.rulebase("ghazal"))["wrong.txt"]
    assert r["pairs"] == 2 and r["edits"] == 3
    assert r["precision"]["exact"] == 1.0 and r["recall"]["exact"] == round(2 / 3, 4)
    assert r["recall_by_edit"]["space>none"]["exact"] == 0.0


def test_unimorph(tmp_path):
    path = tmp_path / "fas"
    path.write_text("خوردن\tمیخورم\tV;1;SG;PRS;IPFV\nخوردن\tخورده بود\tV;3;SG;PST;PRF\n"
                    "آمدن\tبیا\tV;2;SG;IMP\nکار کردن\tکار میکنند\tV;3;PL;PRS;IPFV\n", encoding="utf-8")
    r = verb_eval.unimorph(path, verb.analyzer())
    assert r["simple"] == {"cells": 2, "not_built_cells": 1, "built": 1.0, "right": 1.0, "right_when_built": 1.0,
                           "naive_baseline": 0.5}
    assert r["compound"]["right"] == 1.0
    assert verb_eval.naive("آمدن", mood="imperative", person=2) == "بآم"


def test_g2p_eval(tmp_path):
    path = tmp_path / "bench.csv"
    path.write_text("dataset,grapheme,phoneme,homograph word,pronunciation\n"
                    "homograph,نامه را نوشتم و خانه زیبا بود,nAme rA neveStam va xAne-ye zibA bud,,\n", encoding="utf-8")
    r = g2p_eval.evaluate(path)
    assert r["words_paired"] == round(6 / 7, 4) and r["ezafe_junction"]["sites"] == 2
    assert r["ezafe_junction"]["confusion"] == {"ezafe->ezafe": 1, "none->none": 1}
    assert g2p_eval.fits("گُل", "gol") and not g2p_eval.fits("گِل", "gol") and g2p_eval.fits("خُوبی", "xubi")
    assert g2p_eval.words("کتابِ من", ["ketAb-e", "man"]) == [("کتابِ", "ketAb-e"), ("من", "man")]


def test_aruz_baselines(tmp_path):
    ids = [i for i in range(1, 200) if held_out(str(i))][:2]
    bench, pred = tmp_path / "bench.jsonl", tmp_path / "model.jsonl"
    rows = [{"poem_id": ids[0], "couplet_index": 0, "rhythm": "فاعلاتن فاعلاتن فاعلن",
             "first": "بشنو این نی چون شکایت می‌کند", "second": "از جدایی‌ها حکایت می‌کند"},
            {"poem_id": ids[1], "couplet_index": 0, "rhythm": "فاعلاتن فاعلاتن فاعلن",
             "first": "کز نیستان تا مرا ببریده‌اند", "second": "در نفیرم مرد و زن نالیده‌اند"}]
    bench.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    pred.write_text("".join(json.dumps({"poem_id": r["poem_id"], "couplet_index": 0, "metre": m, "known": True,
                                        "in_training": i == 0}, ensure_ascii=False) + "\n"
                            for i, (r, m) in enumerate(zip(rows, ["فاعلاتن فاعلاتن فاعلن", "فعولن فعولن فعولن فعل"]))),
                    encoding="utf-8")
    r = aruz_baselines.compare(bench, [pred], aruz.scanner())
    assert r["couplets"] == 2
    assert r["model"]["all"]["model"] == 0.5 and r["model"]["known_unseen"]["couplets"] == 1
    assert r["model"]["all"]["majority"] == 1.0
