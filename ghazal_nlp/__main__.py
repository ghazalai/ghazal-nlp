import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from . import aruz, aruz_eval, ganjoor, verb, verb_eval, vowels


def dump(result) -> None:
    print(json.dumps(asdict(result) if is_dataclass(result) else result, ensure_ascii=False, indent=2))


def lines(a) -> list[str]:
    return a.text or sys.stdin.read().splitlines()


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
    scanner = aruz.Scanner(lexicon=None if a.no_lexicon else aruz.WeightLexicon.load(), marks=None if a.stripped else aruz.Marks())
    poems = aruz_eval.load_gold(a.gold)
    report = aruz_eval.evaluate(poems, scanner, split=a.split, limit=a.limit)
    report["readings"] = {name: aruz_eval.score_readings(aruz_eval.load_readings(aruz.DATA / name), scanner)
                          for name in ("readings-thiesen.tsv", "readings-drafted.tsv")}
    dump(report)


def aruz_derive(a):
    poems = aruz_eval.load_gold(a.gold)
    aruz_eval.write_inventory(aruz_eval.build_inventory(poems))
    weights, report = aruz_eval.derive_weights(poems, aruz.Inventory.load())
    aruz_eval.write_weights(weights, report)
    dump(report)


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


def vocalise(a):
    lex = vowels.lexicon()
    for token in vowels.tokens("\n".join(lines(a))):
        e = lex.lookup(token)
        print(token, "unattested" if e is None else " / ".join(f"{r.form}={r.count}" for r in e.readings), sep="\t")


def vowels_derive(a):
    lex, queue, report = vowels.derive(a.gold)
    vowels.write(lex, queue, report)
    dump(report)


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
    c = command("ganjoor", lambda a: [print(json.dumps(x, ensure_ascii=False)) for x in ganjoor.poems(a.snapshot)],
                "poems from a Ganjoor API snapshot, as JSON lines")
    c.add_argument("snapshot", type=Path)
    c = command("aruz-eval", aruz_scores, "score the scanner on the Ganjoor gold set and the reading sets")
    c.add_argument("gold", type=Path, help="the output of `ganjoor`")
    c.add_argument("--split", default="held_out", choices=["held_out", "dev"])
    c.add_argument("--limit", type=int, default=2000)
    c.add_argument("--no-lexicon", action="store_true")
    c.add_argument("--stripped", action="store_true")
    c = command("aruz-derive", aruz_derive, "rebuild the metre inventory and weight lexicon")
    c.add_argument("gold", type=Path)
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
    c = command("vowels", vocalise, "the readings editors wrote for each word")
    c.add_argument("text", nargs="*")
    c = command("vowels-derive", vowels_derive, "rebuild the vowel lexicon and review queue")
    c.add_argument("gold", type=Path, help="the output of `ganjoor`")
    a = p.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
