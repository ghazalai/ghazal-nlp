import json
import sys
from pathlib import Path

from . import checking


def run(args):
    files = args.file or [None]
    if args.file and args.text:
        raise ValueError("use text or --file, not both")
    if getattr(args, "write", False) and (not args.file or "-" in args.file):
        raise ValueError("--write needs a file path; leave it out for text or stdin")
    failed = False
    for name in files:
        if name is None:
            text = "\n".join(args.text) if args.text else sys.stdin.read()
        elif name == "-":
            text = sys.stdin.read()
        else:
            with open(name, encoding="utf-8", newline="") as stream:
                text = stream.read()
        format = args.format or ("markdown" if name and Path(name).suffix.lower() in (".md", ".markdown") else "text")
        options = dict(format=format, style=args.style, registers=tuple(args.register or ["formal_written"]),
                       min_confidence=args.min_confidence, severities=args.severity)
        result = (checking.fix(text, **options) if args.command == "fix"
                  else checking.check(text, legacy=not args.file, **options))
        if args.command == "fix":
            if args.write:
                with open(name, "w", encoding="utf-8", newline="") as stream:
                    stream.write(result["text"])
            if args.json:
                print(json.dumps({"file": name or "-", **result}, ensure_ascii=False))
            elif not args.write:
                print(result["text"], end="")
        else:
            failed |= any(checking.LEVEL[checking.SEVERITY.get(f["severity"], f["severity"])]
                          >= checking.LEVEL[args.fail_on] for f in result["findings"])
            if args.json:
                print(json.dumps({"file": name or "-", **result}, ensure_ascii=False))
            else:
                for f in result["findings"]:
                    if args.file:
                        print(f"{name}:{f['start']}:{f['end']}", f["rule"], f["severity"], f["text"],
                              " / ".join(f["suggestions"]) or "—", sep="\t")
                    else:
                        print(f["rule"] + (" (contested)" if f["contested"] else ""), f["severity"], f["text"],
                              f["suggestion"] or "—", f["citation"], sep="\t")
                for f in result["abstentions"] if args.abstentions else ():
                    print(f["rule_id"], f["text"], f"abstains: {f['reason']}", sep="\t")
    return int(failed)
