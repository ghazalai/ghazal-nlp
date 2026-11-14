from . import aruz, checking, normalize, orthography, verb, vowels


LOCAL = "Runs locally, without a network or key. Uses the package's rules; no spelling tables or hosted checks. "
TEXT = {"type": "string", "maxLength": 20000, "description": "Text to read as data, at most 20,000 characters."}
BOOL = {"type": "boolean"}


def choice(*values):
    return {"type": "string", "enum": list(values)}


def checked(text, format="text", style="ghazal"):
    return {"kind": "formal", "findings": checking.check(text, format=format, style=style)["findings"]}


def fixed(text, format="text", style="ghazal"):
    return {"kind": "formal", **checking.fix(text, format=format, style=style)}


def explain(rule):
    try:
        r = orthography.RuleBase.load().by_id(rule)
    except KeyError:
        raise ValueError("no rule has this id; get the ids from ghazal-nlp rules") from None
    example = next(iter(r.examples_bad), None)
    return {"title": r.title["fa"], "statement": r.statement["fa"],
            "source": {"book": r.source.work, "page": r.source.page} if r.source.page else None,
            "example": {"wrong": example.text, "right": example.expect} if example else None}


def normalized(text, verse=False):
    return {"text": normalize.normalize(text, verse), "normalizer": normalize.VERSION}


def vowelled(text, mode="full", compact=False):
    words = []
    for c in vowels.choose(text):
        readings = vowels.lookup(c.word) if c.confidence < 0.9 else ()
        total = sum(r.count for r in readings)
        words.append({"start": c.span[0], "end": c.span[1], "text": c.word, "form": c.form,
                      "confidence": round(c.confidence, 2), "readings": [
                          {"form": r.form, "weight": round(r.count / total, 2)}
                          for r in readings[:5] if total and r.count / total >= 0.005]})
    marked = vowels.mark(text, mode=mode)
    if compact:
        return {"text": marked, "unsure": [f"@{w['start']} {w['text']}: " + " | ".join(
            f"{r['form']} {r['weight']}" for r in w["readings"]) for w in words if w["readings"]]}
    return {"text": marked, "words": words}


def scanned(text, compact=False):
    lines = aruz.hemistichs(text)[:aruz.MAX_LINES]
    if any(len(line) > 300 for line in lines) or sum(map(len, lines)) > 2400:
        raise ValueError("send at most 2,400 characters of verse, with each hemistich at most 300 characters")
    scanner = aruz.scanner()
    poem = aruz.scan_poem(text, scanner)
    scans = [scanner.scan(line) for line in lines]
    best = next(iter(poem.metres), None)
    pattern = next((p for s in scans for m, p in zip(s.metres, s.patterns) if m == best), None)
    if compact:
        return {"metre": best, "pattern": pattern, "lines": [
            f"{i} " + (f"fits, licences: {dict(s.candidates)[best]}" if best in s.metres
                        else s.abstention or "no reading in this metre") for i, s in enumerate(scans, 1)]}
    return {"written": len(aruz.hemistichs(text)), "best": list(poem.metres), "metres": [
        {"label": m, "pattern": next((p for s in scans for label, p in zip(s.metres, s.patterns) if label == m), None),
         "support": sum(m in s.metres for s in scans),
         "licences": sum(dict(s.candidates).get(m, 0) for s in scans)} for m in poem.metres],
        "lines": [{"text": s.line, "metres": list(s.metres), "abstention": s.abstention} for s in scans]}


def conjugated(text):
    analyzer = verb.analyzer()
    tokens = orthography.tokenize(text)[:40]
    analyses = [analyzer.analyze(t.text) for t in tokens]
    lemmas = dict.fromkeys(r.lemma for a in analyses for r in a.readings)
    if text.strip() in analyzer.stems.by_infinitive:
        lemmas[text.strip()] = None
    tables = {}
    for lemma in lemmas:
        tables[lemma] = {polarity: {
            name: [analyzer.conjugate(lemma, tense=tense, mood=mood, aspect=aspect, polarity=polarity,
                                     person=p, number=n) for n in ("sing", "plur") for p in (1, 2, 3)]
            for name, tense, mood, aspect in (
                ("present", "present", "indicative", None), ("subjunctive", "present", "subjunctive", None),
                ("past", "past", "indicative", None), ("imperfect", "past", "indicative", "imperfective"),
                ("imperative", "present", "imperative", None))} for polarity in ("pos", "neg")}
    return {"verbs": [{"text": t.text, "readings": [
        {**{k: getattr(r, k) for k in ("lemma", "tense", "aspect", "mood", "person", "number", "polarity")},
         "form": r.verb_form} for r in a.readings]} for t, a in zip(tokens, analyses) if a.readings],
        "constructions": [{"words": list(c.tokens), **{k: getattr(c, k) for k in (
            "construction", "lemma", "person", "number", "polarity")}}
            for i in range(len(tokens) - 1) for c in verb.constructions(
                analyses[i], analyses[i + 1], separator=text[tokens[i].end:tokens[i + 1].start])], "tables": tables}


CHECK = {"text": TEXT, "format": choice("text", "markdown"), "style": choice("ghazal", "academy")}
TOOLS = {
    "check_persian": (checked, CHECK, "Check formal Persian prose. Returns findings with rule, start, end, text, suggestions and severity."),
    "fix_persian": (fixed, CHECK, "Apply only errors with one suggestion. Returns text, applied and left. Review the remaining findings."),
    "explain_rule": (explain, {"rule": TEXT}, "Explain a rule by id: title, statement, source and example."),
    "normalize": (normalized, {"text": TEXT, "verse": BOOL}, "Normalize text for storage or comparison. Returns text and normalizer version."),
    "add_vowels": (vowelled, {"text": TEXT, "mode": choice("full", "minimal"), "compact": BOOL}, "Add confident vowel marks. Returns text and words, or text and unsure with compact."),
    "scan_verse": (scanned, {"text": TEXT, "compact": BOOL}, "Scan metre, up to 40 hemistichs. Returns best, metres and lines; compact returns metre, pattern and lines. Rhyme inference and calibrated verse confidence need the hosted API."),
    "conjugate_verb": (conjugated, {"text": TEXT}, "Read verbs in the first 40 words and build conjugation tables. Returns verbs, constructions and tables."),
}


def listed():
    return [{"name": name, "description": LOCAL + about + " Offsets count Unicode code points, end exclusive.",
             "inputSchema": {"type": "object", "properties": fields,
                             "required": ["rule" if name == "explain_rule" else "text"],
                             "additionalProperties": False}} for name, (_, fields, about) in TOOLS.items()]


def call(name, arguments):
    if name not in TOOLS or not isinstance(arguments, dict):
        raise ValueError("send a tool name and an arguments object from tools/list")
    run, fields, _ = TOOLS[name]
    required = "rule" if name == "explain_rule" else "text"
    if required not in arguments or arguments.keys() - fields.keys():
        raise ValueError("send the required field and only fields listed in tools/list")
    for key, value in arguments.items():
        field = fields[key]
        expected = str if field["type"] == "string" else bool
        if type(value) is not expected:
            raise ValueError(f"{key} needs {field['type']}; see tools/list")
        if "maxLength" in field and len(value) > field["maxLength"]:
            raise ValueError("send at most 20,000 characters per call; split the text and try again")
        if "enum" in field and value not in field["enum"]:
            raise ValueError(f"choose {key} from {', '.join(field['enum'])}")
    return run(**arguments)
