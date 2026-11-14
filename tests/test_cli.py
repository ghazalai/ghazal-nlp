import io
import json
import sys

import pytest

from ghazal_nlp import __main__ as cli


def run(monkeypatch, capsys, *args, text=""):
    monkeypatch.setattr(sys, "argv", ["ghazal-nlp", *args])
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    try:
        cli.main()
        status = 0
    except SystemExit as e:
        status = e.code
    out = capsys.readouterr()
    return status, out.out, out.err


@pytest.mark.parametrize("text,extra,status", [
    ("کتاب", [], 0), ("کتاب ها", [], 1), ("آنها", [], 0),
    ("آنها", ["--fail-on", "style"], 1), ("کتاب ها", ["--severity", "house"], 0),
])
def test_gate_exit_codes(monkeypatch, capsys, text, extra, status):
    got, out, err = run(monkeypatch, capsys, "check", text, "--json", *extra)
    assert got == status and not err
    assert json.loads(out)["file"] == "-"


@pytest.mark.parametrize("args", [
    ["check", "--fail-on", "wrong"], ["check", "--file", "missing.md"],
    ["check", "text", "--file", "-"], ["fix", "--file", "-", "--write"],
    ["fix", "text", "--write"], ["conjugate", "گفتن", "--person", "9"],
])
def test_usage_errors_are_one_line_with_a_next_step(monkeypatch, capsys, args):
    status, out, err = run(monkeypatch, capsys, *args)
    assert status == 2 and not out
    assert len(err.splitlines()) == 1 and "--help" in err


def test_bad_utf8_is_a_file_error(monkeypatch, capsys, tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    status, out, err = run(monkeypatch, capsys, "check", "--file", str(path), "--json")
    assert status == 2 and not out and len(err.splitlines()) == 1


def test_markdown_offsets_and_skips(monkeypatch, capsys, tmp_path):
    text = ('---\ntitle: کتاب ها\n---\n```python\nکتاب ها\n```\n'
            '<!-- کتاب ها -->\n[پیوند](https://example.org/كتاب "کتاب ها")\n'
            '[ref]: https://example.org/كتاب\n'
            '`کتاب ها` <a title="کتاب ها">پیوند</a> https://example.org/كتاب\n'
            '# **کتاب ها**\n')
    path = tmp_path / "notes.md"
    path.write_text(text, encoding="utf-8")
    status, out, err = run(monkeypatch, capsys, "check", "--file", str(path), "--json")
    result = json.loads(out)
    assert status == 1 and not err and result["file"] == str(path)
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    start = text.rindex("کتاب ها")
    assert finding == {"rule": "ghazal.plural_space", "start": start, "end": start + 7,
                       "text": "کتاب ها", "suggestions": ["کتاب‌ها"], "severity": "error"}


def test_files_stdin_and_explicit_format(monkeypatch, capsys, tmp_path):
    path = tmp_path / "notes.markdown"
    path.write_text("`کتاب ها`", encoding="utf-8")
    status, out, err = run(monkeypatch, capsys, "check", "--file", str(path), "--file", "-",
                           "--json", text="کتاب ها")
    results = [json.loads(line) for line in out.splitlines()]
    assert status == 1 and not err
    assert [r["file"] for r in results] == [str(path), "-"]
    assert not results[0]["findings"] and results[1]["findings"]
    status, out, _ = run(monkeypatch, capsys, "check", "--format", "markdown", "--json", text="`کتاب ها`")
    assert status == 0 and not json.loads(out)["findings"]
    status, out, _ = run(monkeypatch, capsys, "check", "--file", str(path), "--format", "text", "--json")
    assert status == 1 and json.loads(out)["findings"]


def test_fix_round_trip_and_original_offsets(monkeypatch, capsys, tmp_path):
    original = "```\nکتاب ها\n```\nاین كتاب ها را ميخوانم .\nآنها\n"
    path = tmp_path / "notes.md"
    path.write_text(original, encoding="utf-8")
    status, out, err = run(monkeypatch, capsys, "fix", "--file", str(path), "--json")
    result = json.loads(out)
    assert status == 0 and not err and result["applied"]
    assert path.read_text(encoding="utf-8") == original
    assert result["text"].startswith("```\nکتاب ها\n```\n")
    assert result["text"].endswith("آنها\n")
    for f in result["applied"]:
        assert original[f["start"]:f["end"]] == f["text"]
        assert f["severity"] == "error" and len(f["suggestions"]) == 1
    replay = original
    for f in reversed(result["applied"]):
        replay = replay[:f["start"]] + f["suggestions"][0] + replay[f["end"]:]
    assert replay == result["text"]
    run(monkeypatch, capsys, "fix", "--file", str(path), "--write")
    assert path.read_text(encoding="utf-8") == result["text"]
    status, out, _ = run(monkeypatch, capsys, "fix", "--file", str(path), "--write", "--json")
    second = json.loads(out)
    assert second["text"] == result["text"] and not second["applied"]
    status, out, _ = run(monkeypatch, capsys, "check", "--file", str(path), "--json")
    assert status == 0


def test_crlf_offsets_and_fix_preserve_newlines(monkeypatch, capsys, tmp_path):
    original = "---\r\ntitle: کتاب ها\r\n---\r\n```\r\nکتاب ها\r\n```\r\nکتاب ها\r\n"
    path = tmp_path / "notes.md"
    path.write_bytes(original.encode("utf-8"))
    status, out, _ = run(monkeypatch, capsys, "check", "--file", str(path), "--json")
    findings = json.loads(out)["findings"]
    assert status == 1 and len(findings) == 1
    assert findings[0]["start"] == original.rindex("کتاب ها")
    run(monkeypatch, capsys, "fix", "--file", str(path), "--write")
    assert path.read_bytes().decode("utf-8") == original[:original.rindex("کتاب ها")] + "کتاب‌ها\r\n"


@pytest.mark.parametrize("args", [
    ["scan", "باغ", "راه", "--no-lexicon"], ["scan", "باغ", "--poem", "--no-lexicon"],
    ["rhyme", "باغ\nداغ\nراه\nماه", "--form", "masnavi"], ["verb", "گفت رفت"],
    ["conjugate", "گفتن"], ["conjugate", "ناشناخته"], ["vowels", "باغ راه"],
    ["ezafe", "خانه زیبا"], ["ezafe-heh", "خانه زیبا"], ["arabic", "باغ"],
    ["rules"], ["verify"], ["normalize", "كتاب"], ["check", "کتاب"], ["fix", "کتاب"],
    ["vowels", ""], ["verb", ""], ["ezafe", ""], ["ezafe-heh", ""], ["scan", "", "--no-lexicon"],
])
def test_json_is_one_parseable_value(monkeypatch, capsys, args):
    status, out, err = run(monkeypatch, capsys, *args, "--json")
    assert status == 0 and not err
    json.loads(out)


def test_already_json_commands_accept_the_flag(monkeypatch, capsys, tmp_path):
    for command in ("aruz-eval", "aruz-derive", "rhyme-eval", "verb-eval", "vowels-derive",
                    "ezafe-eval", "profile", "arabic-eval", "aruz-baselines", "conjugate-eval",
                    "g2p-eval", "orthography-eval", "shahnameh-eval", "ganjoor"):
        status, out, err = run(monkeypatch, capsys, command, "--json", "--help")
        assert status == 0 and "--json" in out and not err
    path = tmp_path / "notes.txt"
    path.write_text("باغ", encoding="utf-8")
    status, out, err = run(monkeypatch, capsys, "profile", str(path), "--json")
    assert status == 0 and json.loads(out)["documents"] == 1


@pytest.mark.parametrize("command,arguments,module,function", [
    ("rhyme-eval", ["made-up.jsonl"], "rhyme_eval", "evaluate"),
    ("verb-eval", ["made-up.conllu"], "verb_eval", "evaluate"),
    ("arabic-eval", [], "arabic_eval", "evaluate"),
    ("aruz-baselines", ["made-up.jsonl", "predictions.jsonl"], "aruz_baselines", "compare"),
    ("conjugate-eval", ["made-up.tsv"], "verb_eval", "unimorph"),
    ("g2p-eval", ["made-up.csv"], "g2p_eval", "evaluate"),
    ("orthography-eval", ["wrong.txt", "right.txt"], "orthography_eval", "evaluate"),
    ("shahnameh-eval", [], "shahnameh_eval", "evaluate"),
])
def test_evaluation_json_with_a_tiny_report(monkeypatch, capsys, command, arguments, module, function):
    monkeypatch.setattr(getattr(cli, module), function, lambda *a, **kw: {"examples": 1})
    status, out, err = run(monkeypatch, capsys, command, *arguments, "--json")
    assert status == 0 and not err and json.loads(out) == {"examples": 1}


def test_ganjoor_json_lines(monkeypatch, capsys, tmp_path):
    (tmp_path / "cat").mkdir()
    poems = [{"id": 1, "text": "باغ"}, {"id": 2, "text": "راه"}]
    monkeypatch.setattr(cli.ganjoor, "poems", lambda path: iter(poems))
    status, out, err = run(monkeypatch, capsys, "ganjoor", str(tmp_path), "--json")
    assert status == 0 and not err and [json.loads(line) for line in out.splitlines()] == poems


@pytest.mark.parametrize("command", ["aruz-eval", "aruz-derive", "vowels-derive", "ezafe-eval"])
def test_remaining_reports_are_json(monkeypatch, capsys, tmp_path, command):
    from pathlib import Path
    report = {"examples": 1}
    arguments = []
    if command.startswith("aruz-"):
        monkeypatch.setattr(cli.aruz_eval, "load_gold", lambda *a, **kw: [])
        arguments = [str(tmp_path / "poems.jsonl")]
        if command == "aruz-eval":
            monkeypatch.setattr(cli.aruz_eval, "evaluate", lambda *a, **kw: report)
            monkeypatch.setattr(cli.aruz_eval, "load_readings", lambda *a: [])
            monkeypatch.setattr(cli.aruz_eval, "score_readings", lambda *a: {"examples": 1})
            arguments += ["--no-lexicon"]
        else:
            monkeypatch.setattr(cli.aruz_eval, "build_inventory", lambda *a: [])
            monkeypatch.setattr(cli.aruz_eval, "write_inventory", lambda *a: None)
            monkeypatch.setattr(cli.aruz.Inventory, "load", lambda *a: None)
            monkeypatch.setattr(cli.aruz_eval, "derive_weights", lambda *a: ({}, report))
            monkeypatch.setattr(cli.aruz_eval, "write_weights", lambda *a: None)
            arguments += ["--out", str(tmp_path)]
    elif command == "vowels-derive":
        monkeypatch.setattr(cli.vowels, "derive", lambda *a: ({}, [], report))
        monkeypatch.setattr(cli.vowels, "write", lambda *a: None)
        arguments = [str(tmp_path / "poems.jsonl"), "--out", str(tmp_path)]
    else:
        for name in ("ezafe-written-dev.jsonl", "ezafe-junctions-dev.jsonl", "ezafe-lab-1.jsonl"):
            Path(tmp_path / name).write_text('{"observed_mark_class":"unmarked"}\n', encoding="utf-8")
        monkeypatch.setattr(cli.aruz, "DATA", tmp_path)
        monkeypatch.setattr(cli.ezafe, "score", lambda *a: report)
    status, out, err = run(monkeypatch, capsys, command, *arguments, "--json")
    assert status == 0 and not err and isinstance(json.loads(out), dict)
