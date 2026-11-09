import json
import sys
from types import SimpleNamespace

import pytest

from ghazal_nlp import __main__ as cli, orthography as o, orthography_eval as ev


def test_function_splits_are_active_low_confidence_advisories():
    base = o.rulebase()
    rule = base.by_id("ghazal.function_splits")
    assert rule.runs and (rule.severity, rule.confidence) == ("advisory", "low")
    assert not base.check("بهروش").findings
    findings = base.check("بهروش", min_confidence="low").findings
    assert [(f.rule_id, f.suggestion, f.severity, f.confidence) for f in findings] == [
        (rule.id, "به روش", "advisory", "low")]


@pytest.mark.parametrize("severities", [(), ("advisory",), ("violation",), ("house",), ("violation", "house")])
def test_check_filters_severities(severities):
    base = o.rulebase()
    text = "بهروش کتاب ها کتــاب"
    all_findings = base.check(text, min_confidence="low").findings
    assert {f.severity for f in all_findings} == o.SEVERITIES
    got = base.check(text, min_confidence="low", severities=severities).findings
    assert got == tuple(f for f in all_findings if f.severity in severities)
    assert not base.check("بهروش", severities=("advisory",)).findings


def test_check_rejects_unknown_severity():
    with pytest.raises(ValueError, match="unknown severities"):
        o.rulebase().check("متن", severities=("warning",))


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_reports_advisory_without_calling_it_a_violation(monkeypatch, capsys, as_json):
    argv = ["ghazal-nlp", "check", "بهروش کتاب ها کتــاب", "--min-confidence", "low", "--severity", "advisory"]
    monkeypatch.setattr(sys, "argv", argv + (["--json"] if as_json else []))
    cli.main()
    output = capsys.readouterr().out
    if as_json:
        findings = json.loads(output)["findings"]
        assert len(findings) == 1
        assert (findings[0]["severity"], findings[0]["confidence"]) == ("advisory", "low")
    else:
        assert output.startswith("ghazal.function_splits\tadvisory\tبهروش\tبه روش\t")
        assert len(output.splitlines()) == 1


def test_eval_reports_advisory_separately(tmp_path):
    base = o.rulebase()
    include_low = SimpleNamespace(check=lambda text, **kw: base.check(text, min_confidence="low", **kw),
                                  for_registers=base.for_registers)
    wrong, correct = tmp_path / "wrong.txt", tmp_path / "correct.txt"
    wrong.write_text("بهروش\n"); correct.write_text("به روش\n")
    report = ev.evaluate([(wrong, correct)], include_low)[wrong.name]
    assert report["by_severity"] == {"advisory": {"changes": 1, "exact": 1.0, "apart": 1.0}}
    assert report["by_rule"]["ghazal.function_splits"]["counts"]["changes"] == 1
