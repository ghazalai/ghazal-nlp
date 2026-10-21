import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from . import arabic, arabic_eval, aruz, aruz_eval, ezafe, ganjoor, orthography, rhyme, rhyme_eval, verb, verb_eval, vowels
from . import aruz_baselines, g2p_eval, orthography_eval, shahnameh_eval


def dump(result) -> None:
    print(json.dumps(asdict(result) if is_dataclass(result) else result, ensure_ascii=False, indent=2))


def lines(a) -> list[str]:
    return a.text or sys.stdin.read().splitlines()


def snapshot(a):
    if not (a.snapshot / "cat").is_dir():
        sys.exit(f"ghazal-nlp: {a.snapshot} is not a Ganjoor API snapshot (no cat/ folder)")
    for poem in ganjoor.poems(a.snapshot):
        print(json.dumps(poem, ensure_ascii=False))


def scan(a):
    scanner = aruz.Scanner(lexicon=None if a.no_lexicon else aruz.WeightLexicon.load(), marks=None if a.stripped else aruz.Marks())
    if a.poem:
        result = aruz.scan_poem("\n".join(lines(a)), scanner)
        return dump(result) if a.json else print(" | ".join(result.metres) or "abstains", f"· {result.support} of {result.lines} hemistichs")
    for line in aruz.hemistichs("\n".join(lines(a))):
        s = scanner.scan(line)
        if a.json:
            dump(s)
            continue
        print(line, "\n  ", " | ".join(s.metres) or f"abstains: {s.abstention}")
        if s.metres:
            print("  ", " ".join(s.weights), "·", ", ".join(s.devices) or "no licences")


def aruz_scores(a):
    lexicon = None if a.no_lexicon else aruz.WeightLexicon.load(a.weights) if a.weights else aruz.WeightLexicon.load()
    scanner = aruz.Scanner(lexicon=lexicon, marks=None if a.stripped else aruz.Marks())
    poems = aruz_eval.load_gold(a.gold, by=a.by)
    report = aruz_eval.evaluate(poems, scanner, split=a.split, limit=a.limit)
    report["thiesen"] = aruz_eval.score_readings(aruz_eval.load_readings(aruz.DATA / "readings-thiesen.tsv"), scanner)
    dump(report)


def aruz_derive(a):
    poems = aruz_eval.load_gold(a.gold, by=a.by)
    aruz_eval.write_inventory(aruz_eval.build_inventory(poems), a.out / "metres.tsv")
    weights, report = aruz_eval.derive_weights(poems, aruz.Inventory.load(a.out / "metres.tsv"))
    aruz_eval.write_weights(weights, report, a.out / "weights.tsv")
    dump(report)


def rhymes(a):
    for r in rhyme.scan(lines(a), a.form):
        if a.json:
            dump(r)
        elif r.abstention:
            print("abstains:", r.abstention)
        else:
            print(f"qafiya {r.qafiya} · radif {r.radif or '—'} · rhyme words {'، '.join(r.words)}"
                  + (f" · off: {', '.join(str(i + 1) for i in r.off)}" if r.off else "")
                  + (f" · repeated: {'، '.join(r.repeated)}" if r.repeated else ""))


def analyse(a):
    for token in " ".join(lines(a)).split():
        result = verb.analyzer().analyze(token)
        if a.json:
            dump(result)
            continue
        print(token)
        for r in result.readings:
            print("  ", r.lemma, r.features(), "·", ", ".join(r.rules) or "inflection")
        if not result.readings:
            print("   abstains:", result.abstention)


def conjugate(a):
    print(verb.analyzer().conjugate(a.infinitive, tense=a.tense, mood=a.mood, person=a.person, number=a.number,
                                    polarity=a.polarity, aspect=a.aspect) or "no such form")


def vocalize(a):
    lex = vowels.lexicon()
    for token in vowels.tokens("\n".join(lines(a))):
        e = lex.lookup(token)
        print(token, "unattested" if e is None else " / ".join(f"{r.form}={r.count}" for r in e.readings), sep="\t")


def vowels_derive(a):
    lex, queue, report = vowels.derive(a.gold)
    vowels.write(lex, queue, report, a.out)
    dump(report)


def junctions(a):
    for line in lines(a):
        for j in ezafe.junctions(line, a.profile, a.metre):
            print(line[slice(*j.span)], j.mark, j.label, j.reason, sep="\t")


def ezafe_scores(a):
    report = {}
    for name, classify in (("ezafe-written-dev.jsonl", ezafe.written), ("ezafe-junctions-dev.jsonl", ezafe.junction)):
        rows = [json.loads(line) for line in open(aruz.DATA / name, encoding="utf-8")]
        report[name] = {t: ezafe.score(rows, classify, t) for t in ("blind", "revised")}
    lab = [json.loads(line) for line in open(aruz.DATA / "ezafe-lab-1.jsonl", encoding="utf-8")]
    report["ezafe-lab-1.jsonl"] = {
        "written": ezafe.score([r for r in lab if r["observed_mark_class"] != "unmarked"], ezafe.written, "blind"),
        "junction": ezafe.score([r for r in lab if r["observed_mark_class"] == "unmarked"], ezafe.junction, "blind")}
    dump(report)


def rules(a):
    base = orthography.rulebase(a.style)
    if a.json:
        return dump(base.summary())
    for rule in base.rules:
        state = "runs" if rule.runs else "recorded" if not rule.executable else "off"
        where = f"ص {rule.source.page}" if rule.source.page else f"academy: {rule.academy['position']}"
        print(rule.id, rule.status, ",".join(rule.registers), state, where, sep="\t")
    s = base.summary()
    print(f"recorded {s['recorded']} · executable {s['executable']} · running {s['running']}")


def check(a):
    analysis = orthography.rulebase(a.style).check("\n".join(lines(a)), registers=tuple(a.register or ["formal_written"]),
                                                   min_confidence=a.min_confidence)
    if a.json:
        return dump(analysis)
    for x in analysis.findings:
        print(x.rule_id + (" (contested)" if x.contested else ""), x.severity, x.text, x.suggestion or "—",
              x.citation, sep="\t")
    for x in analysis.abstentions if a.abstentions else ():
        print(x.rule_id, x.text, f"abstains: {x.reason}", sep="\t")


def verify(a):
    try:
        base = orthography.RuleBase.load()
    except orthography.RuleError as e:
        sys.exit(f"rule base failed verification\n{e}")
    print(f"{len(base.rules)} rules pass their own examples")


def profile(a):
    files = sorted(a.path.glob("*.txt")) if a.path.is_dir() else [a.path]
    dump(orthography.rulebase(a.style).profile((f.read_text(encoding="utf-8") for f in files),
                                        registers=tuple(a.register or ["formal_written"])))


def passages(a):
    text = "\n".join(lines(a))
    spans = arabic.detector().detect(text) + (arabic.QuranIndex(a.quran).detect(text) if a.quran else [])
    if a.json:
        return dump([asdict(s) for s in spans])
    for s in sorted(spans, key=lambda s: s.start):
        print(s.start, s.end, text[s.start:s.end], ",".join(s.reason + s.quran_refs), sep="\t")


def main() -> None:
    p = argparse.ArgumentParser(prog="ghazal-nlp")
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, run, help):
        c = sub.add_parser(name, help=help)
        c.set_defaults(run=run)
        return c

    c = command("scan", scan, "the metre of hemistichs, or of a whole poem")
    c.add_argument("text", nargs="*", help="hemistichs; stdin when none")
    c.add_argument("--poem", action="store_true", help="one metre for all the lines")
    c.add_argument("--no-lexicon", action="store_true", help="the unsupervised scanner")
    c.add_argument("--stripped", action="store_true", help="ignore the editor's marks")
    c.add_argument("--json", action="store_true")
    c = command("ganjoor", snapshot, "poems from a Ganjoor API snapshot, as JSON lines")
    c.add_argument("snapshot", type=Path)
    c = command("aruz-eval", aruz_scores, "score the scanner on the Ganjoor gold set and the reading sets")
    c.add_argument("gold", type=Path, help="the output of `ganjoor`")
    c.add_argument("--split", default="held_out", choices=["held_out", "dev"])
    c.add_argument("--limit", type=int, default=2000)
    c.add_argument("--by", default="poem", choices=["poem", "poet"], help="hold out poems or whole poets")
    c.add_argument("--weights", type=Path, help="a weights.tsv from aruz-derive --by poet")
    c.add_argument("--no-lexicon", action="store_true")
    c.add_argument("--stripped", action="store_true")
    c = command("aruz-derive", aruz_derive, "rebuild metres.tsv and weights.tsv")
    c.add_argument("gold", type=Path)
    c.add_argument("--out", type=Path, default=Path("."), help="where to write them")
    c.add_argument("--by", default="poem", choices=["poem", "poet"])
    c = command("rhyme", rhymes, "the rhyme letters and radif of a poem, one hemistich per line")
    c.add_argument("text", nargs="*")
    c.add_argument("--form", required=True, choices=rhyme.FORMS)
    c.add_argument("--json", action="store_true")
    c = command("rhyme-eval", lambda a: dump(rhyme_eval.evaluate(a.gold, a.split, misses=a.misses)),
                "score the rhyme finder against Ganjoor's rhyme letters")
    c.add_argument("gold", type=Path, help="the output of `ganjoor`")
    c.add_argument("--split", default="held_out", choices=["held_out", "dev"])
    c.add_argument("--misses", type=int, default=0, help="show this many disagreements")
    c = command("verb", analyse, "every reading of verb tokens")
    c.add_argument("text", nargs="*")
    c.add_argument("--json", action="store_true")
    c = command("conjugate", conjugate, "one form of a verb")
    c.add_argument("infinitive")
    c.add_argument("--tense", default="present", choices=["present", "past"])
    c.add_argument("--mood", default="indicative", choices=["indicative", "subjunctive", "imperative"])
    c.add_argument("--person", type=int, default=3)
    c.add_argument("--number", default="sing", choices=["sing", "plur"])
    c.add_argument("--polarity", default="pos", choices=["pos", "neg"])
    c.add_argument("--aspect", choices=["imperfective", "imperfective_perfect"])
    c = command("verb-eval", lambda a: dump(verb_eval.evaluate(a.conllu, verb.analyzer())),
                "score the verb analyser on UD treebank files")
    c.add_argument("conllu", nargs="+", type=Path)
    c = command("vowels", vocalize, "the readings editors wrote for each word")
    c.add_argument("text", nargs="*")
    c = command("vowels-derive", vowels_derive, "rebuild vowels-lexicon.tsv and vowels-review.tsv")
    c.add_argument("gold", type=Path, help="the output of `ganjoor`")
    c.add_argument("--out", type=Path, default=Path("."), help="where to write them")
    c = command("ezafe", junctions, "each silent-ه junction and whether it carries an ezāfe")
    c.add_argument("text", nargs="*")
    c.add_argument("--profile", default="prose", choices=["prose", "verse"])
    c.add_argument("--metre", help="the poem's metre, as Ganjoor labels it, for --profile verse")
    command("ezafe-eval", ezafe_scores, "score the ezāfe classifiers on my labels")
    registers = sorted(orthography.REGISTERS)
    c = command("rules", rules, "the spelling rules: what is recorded, what runs")
    c.add_argument("--style", default="ghazal", choices=orthography.STYLES)
    c.add_argument("--json", action="store_true")
    c = command("check", check, "check text against the spelling rules")
    c.add_argument("text", nargs="*")
    c.add_argument("--style", default="ghazal", choices=orthography.STYLES)
    c.add_argument("--register", action="append", choices=registers, help="formal_written when none")
    c.add_argument("--min-confidence", default="medium", choices=["low", "medium", "high"])
    c.add_argument("--abstentions", action="store_true", help="also what the rules declined to judge")
    c.add_argument("--json", action="store_true")
    command("verify", verify, "run every rule against its own examples")
    c = command("profile", profile, "rule coverage over a file or a directory of .txt files")
    c.add_argument("path", type=Path)
    c.add_argument("--style", default="ghazal", choices=orthography.STYLES)
    c.add_argument("--register", action="append", choices=registers)
    c = command("arabic", passages, "Arabic passages inside Persian")
    c.add_argument("text", nargs="*")
    c.add_argument("--quran", type=Path, action="append", default=[], help="a Tanzil copy, sura|aya|text lines")
    command("arabic-eval", lambda a: dump(arabic_eval.evaluate()), "score the Arabic detector on my labels")
    c.add_argument("--json", action="store_true")
    c = command("aruz-baselines", lambda a: dump(aruz_baselines.compare(a.bench, a.predictions, aruz.scanner())),
                "the scanner next to other metre classifiers on Ganjoor-Rhythm-Bench's held-out couplets")
    c.add_argument("bench", type=Path, help="Ganjoor-Rhythm-Bench's benchmark.jsonl")
    c.add_argument("predictions", nargs="+", type=Path, help="a model's predictions, JSON lines")
    c = command("conjugate-eval", lambda a: dump(verb_eval.unimorph(a.unimorph, verb.analyzer())),
                "score conjugate on UniMorph's Persian paradigms")
    c.add_argument("unimorph", type=Path, help="UniMorph's fas file")
    c = command("g2p-eval", lambda a: dump(g2p_eval.evaluate(a.csv)),
                "score ezafe and vowels on SentenceBench or HomoRich pronunciations")
    c.add_argument("csv", type=Path, help="SentenceBench.csv or HomoRich's test_set.csv")
    c = command("orthography-eval", lambda a: dump(orthography_eval.evaluate(
        list(zip(a.files[::2], a.files[1::2])), orthography.rulebase(a.style), limit=a.limit)),
        "score the space and ZWNJ rules on PerSpellData's word-boundary pairs")
    c.add_argument("files", nargs="+", type=Path, help="wrong and correct file, in pairs")
    c.add_argument("--style", default="ghazal", choices=orthography.STYLES)
    c.add_argument("--limit", type=int, help="pairs per file")
    command("shahnameh-eval", lambda a: dump(shahnameh_eval.evaluate()),
            "score vowels and ezafe against the Tajik Shahnameh")
    a = p.parse_args()
    try:
        a.run(a)
    except OSError as e:
        sys.exit(f"ghazal-nlp: {e}")


if __name__ == "__main__":
    main()
