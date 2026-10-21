"""The Ganjoor gold set: the metre inventory and weight lexicon derived from its
development split, and the scanner scored on its held-out fifth."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

from .aruz import (
    CAESURA, CONFLICT, COST, DATA, LONG, MAX_LINES, OVERLONG, PINNED, RUBAI, SHORT, UNDERDETERMINED, VARIABLE,
    Devices, Inventory, MetreEntry, Scanner, ScriptError, Trie, WeightEntry, _constrainable_words,
    _glide, _glide_weights, _moves, _pattern_slots, _slot_blocked, _word_spans, hemistichs,
    pattern_for_label, prepare, scan_poem, votes, winners,
)

SNAPSHOT = "Ganjoor editorial metre labels, snapshot api-2026-08-03"


@dataclass(frozen=True, slots=True)
class GoldPoem:
    id: str
    metre: str
    text: str
    held_out: bool


def held_out(poem_id: str) -> bool:
    """A fixed fifth of the poems, by id. The hash chain is the one every
    published number used."""
    doc_id = hashlib.sha1(f"poem\x1fganjoor\x1f{poem_id}".encode()).hexdigest()
    return int.from_bytes(hashlib.blake2b(doc_id.encode(), digest_size=8).digest(), "big") % 5 == 0


def load_gold(path: Path) -> list[GoldPoem]:
    """Poems with a single metre label, from ``ganjoor.poems`` written as JSON lines."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["metre_status"] == "single" and p["text"] and p["metre"]:
                out.append(GoldPoem(p["id"], tidy(p["metre"]), p["text"], held_out(p["id"])))
    named = Counter(p.metre for p in out if not p.held_out and "(" in p.metre)
    names = {}
    for label, _ in named.most_common():
        names.setdefault(label.split(" (")[0], label)
    return [replace(p, metre=names.get(p.metre, p.metre)) for p in out]


def tidy(label: str) -> str:
    """One space between words and before the name."""
    return re.sub(r"\s*\(", " (", " ".join(label.split()))


def rubai_variants(pattern: str) -> list[str]:
    """The robāʿī family: the middle foot alternates –⏑⏑– with ⏑–⏑–, and any ⏑⏑
    may fuse into one long (Thiesen §§156, 255–256)."""
    variants = {pattern}
    if pattern[5:9] == "-uu-":
        variants.add(pattern[:5] + "u-u-" + pattern[9:])
    for _ in range(pattern.count("uu")):
        for base in list(variants):
            start = 0
            while (i := base.find("uu", start)) >= 0:
                variants.add(base[:i] + LONG + base[i + 2:])
                start = i + 1
    variants.discard(pattern)
    return sorted(variants)


def build_inventory(poems) -> list[MetreEntry]:
    counts = Counter(p.metre for p in poems if not p.held_out)
    entries = []
    for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pattern = pattern_for_label(label)
        if pattern is None:
            entries.append(MetreEntry(label, "", "unexpanded", n))
            continue
        entries.append(MetreEntry(label, pattern, "arkan", n))
        if RUBAI in label:
            entries += [MetreEntry(label, v, "derived", n) for v in rubai_variants(pattern)]
    return entries


def write_inventory(entries, path: Path = DATA / "metres.tsv") -> None:
    header = [
        f"# Persian metres. Source: {SNAPSHOT}.",
        "# One row per label and weight pattern (u short, - long). `arkan`: the label's feet",
        "# expanded; `derived`: a robāʿī variant; `unexpanded`: a foot missing from aruz.FEET.",
        "# label\tpattern\tprovenance\tdev_poems",
    ]
    rows = [f"{e.label}\t{e.pattern}\t{e.provenance}\t{e.poems}" for e in entries if e.pattern or e.provenance == "unexpanded"]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


ACCEPT = -1
MAX_WORD_READINGS = 32
COST_MARGIN = 2  # a word's reading counts only within this many licences of the line's cheapest
NO_COST = 1 << 30


def _live_graph(skel, trie: Trie, devices: Devices):
    """The transitions on some complete scansion, each with its own weight and
    cost, and the cheapest cost to reach (``fcost``) and complete (``bcost``) each state."""
    n = len(skel)
    forward = [set() for _ in range(n + 1)]
    forward[0].add(0)
    fcost = [{} for _ in range(n + 1)]
    fcost[0][0] = 0
    edges = [[] for _ in range(n)]
    glides = [[] for _ in range(n)]

    def relax(store, node, cost):
        if cost < store.get(node, NO_COST):
            store[node] = cost

    for i in range(n):
        nodes = forward[i]
        if not nodes:
            continue
        fc = fcost[i]
        if _glide(skel, i, devices):
            for node, base in [(node, fc[node]) for node in nodes]:
                for weight, dev in _glide_weights(devices):
                    landed = trie.walk(node, weight)
                    if landed is not None:
                        glides[i].append((node, landed, COST[dev]))
                        nodes.add(landed)
                        relax(fc, landed, base + COST[dev])
        if trie.caesura and i and skel.word[i] != skel.word[i - 1]:
            for node, base in [(node, fc[node]) for node in nodes]:
                half = trie.children[node].get(CAESURA)
                if half is not None:
                    glides[i].append((node, half, 0))
                    nodes.add(half)
                    relax(fc, half, base)
        seen = {}

        def add(edge, cost):
            at = seen.get(edge)
            if at is None:
                seen[edge] = len(edges[i])
                edges[i].append((*edge, cost))
            elif cost < edges[i][at][4]:
                edges[i][at] = (*edge, cost)

        for weights, nxt, dev in _moves(skel, i, devices):
            for realized, licence in _pattern_slots(weights, i, devices):
                extra = COST[dev] + COST[licence]
                for node in nodes:
                    if licence and _slot_blocked(licence, trie.depth[node], devices):
                        continue
                    landed = trie.walk(node, realized)
                    if landed is not None:
                        forward[nxt].add(landed)
                        relax(fcost[nxt], landed, fc[node] + extra)
                        add((node, landed, nxt, weights), extra)
                    if nxt != n:
                        if trie.caesura and skel.word[nxt] != skel.word[nxt - 1]:
                            spilled = trie.spill(node, realized)
                            if spilled is not None:
                                forward[nxt].add(spilled)
                                relax(fcost[nxt], spilled, fc[node] + extra)
                                add((node, spilled, nxt, weights), extra)
                        continue
                    if any(trie.terminal[c] is not None and c != landed for c in trie.children[node].values()):
                        add((node, ACCEPT, n, weights), extra)
    complete = [set() for _ in range(n + 1)]
    complete[n] = {node for node in forward[n] if trie.terminal[node] is not None} | {ACCEPT}
    bcost = [{} for _ in range(n + 1)]
    bcost[n] = {node: 0 for node in complete[n]}
    for i in range(n - 1, -1, -1):
        bc = bcost[i]
        for node, landed, nxt, _, cost in edges[i]:
            if landed in complete[nxt]:
                complete[i].add(node)
                relax(bc, node, cost + bcost[nxt][landed])
        for node, landed, g in reversed(glides[i]):
            if landed in complete[i]:
                complete[i].add(node)
                relax(bc, node, g + bc[landed])
    return forward, complete, edges, glides, fcost, bcost


def word_weights(skel, trie: Trie, devices: Devices) -> dict[int, frozenset[str]] | None:
    """What each word could have weighed in a line that fills ``trie``; None if it cannot."""
    forward, complete, edges, glides, fcost, bcost = _live_graph(skel, trie, devices)
    if 0 not in complete[0]:
        return None
    ceiling = bcost[0][0] + COST_MARGIN
    eligible = _constrainable_words(skel, devices)
    out = {}
    for wid, (start, end) in enumerate(_word_spans(skel)):
        if not eligible[wid]:
            continue
        states = [{} for _ in range(end - start + 1)]
        states[0] = {node: {"": fcost[start][node]} for node in forward[start] if node in complete[start]}
        for node, landed, g in glides[start]:  # the previous word's ezāfe
            if node in states[0] and landed in complete[start]:
                into = states[0].setdefault(landed, {})
                for w, c in states[0][node].items():
                    if c + g < into.get(w, NO_COST):
                        into[w] = c + g
        crossed = False
        for offset in range(end - start):
            here = states[offset]
            for node, landed, nxt, weights, cost in edges[start + offset] if here else ():
                if node not in here or landed not in complete[nxt]:
                    continue
                if nxt > end:
                    crossed = True
                    break
                grown = states[nxt - start].setdefault(landed, {})
                for w, c in here[node].items():
                    if c + cost < grown.get(w + weights, NO_COST):
                        grown[w + weights] = c + cost
                if len(grown) > MAX_WORD_READINGS:
                    crossed = True
                    break
            if crossed:
                break
        if crossed:
            continue
        found = {}
        for node, ends in states[end - start].items():
            for w, c in ends.items():
                if c + bcost[end][node] < found.get(w, NO_COST):
                    found[w] = c + bcost[end][node]
        if found := {w for w, c in found.items() if c <= ceiling}:
            out[wid] = frozenset(found)
    return out


def classify(counter: dict[frozenset[str], int], max_weights: int, coverage: float):
    """One word's status, the weights it stands behind, and its evidence.

    pinned: one string fits ~every occurrence. underdetermined: several do.
    variable: none does, but a small set covers them (به is /be/ and /beh/).
    conflict: no small set covers them.
    """
    occurrences = sum(counter.values())
    tally = Counter()
    for admissible, count in counter.items():
        for w in admissible:
            tally[w] += count
    support = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    threshold = max(1, min(occurrences, -(-occurrences * round(coverage * 10_000) // 10_000)))
    surviving = tuple(sorted(w for w, c in support if c >= threshold))
    if len(surviving) == 1:
        return PINNED, surviving, occurrences, support
    if surviving:
        return UNDERDETERMINED, surviving if len(surviving) <= max_weights else (), occurrences, support
    uncovered, remaining, cover = dict(counter), occurrences, []
    while remaining > occurrences - threshold and len(cover) < max_weights:
        tally = Counter()
        for admissible, count in uncovered.items():
            for w in admissible:
                tally[w] += count
        if not tally:
            break
        best = min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        cover.append(best)
        uncovered = {a: c for a, c in uncovered.items() if best not in a}
        remaining = sum(uncovered.values())
    if remaining > occurrences - threshold:
        return CONFLICT, (), occurrences, support
    return VARIABLE, tuple(sorted(cover)), occurrences, support


def derive_weights(poems, inventory: Inventory, devices: Devices = Devices(), *, max_lines: int = 24,
                   record_min: int = 2, max_weights: int = 6, coverage: float = 0.99):
    """The weight lexicon from the development split only: the label fixes the
    pattern, and the scanner reads off what each word must have weighed."""
    tries, stats, seen = {}, Counter(poems=0, lines=0, lines_used=0), {}
    for gold in poems:
        if gold.held_out:
            continue
        patterns = inventory.label_patterns.get(gold.metre)
        if not patterns:
            stats["poems_label_unknown"] += 1
            continue
        if gold.metre not in tries:
            tries[gold.metre] = Trie(sorted(set(patterns)), inventory.caesuras if devices.caesura else None)
        stats["poems"] += 1
        for line in hemistichs(gold.text)[:max_lines]:
            stats["lines"] += 1
            try:
                skel = prepare(line)
            except ScriptError:
                stats["lines_not_script"] += 1
                continue
            observed = word_weights(skel, tries[gold.metre], devices)
            if observed is None:
                stats["lines_off_label"] += 1
                continue
            stats["lines_used"] += 1
            for wid, weights in observed.items():
                seen.setdefault(skel.words[wid], Counter())[weights] += 1
    entries, outcomes = [], Counter()
    for word, counter in seen.items():
        status, weights, occurrences, support = classify(counter, max_weights, coverage)
        outcomes[status] += 1
        outcomes[f"{status}_occurrences"] += occurrences
        if occurrences >= record_min:
            hit = sum(c for a, c in counter.items() if a & set(weights))
            agreement = round(hit / occurrences, 4) if weights else 0.0
            entries.append(WeightEntry(word, status, weights, occurrences, agreement, tuple(support[:6])))
    entries.sort(key=lambda e: (-e.occurrences, e.word))
    return entries, {**stats, "distinct_words": len(seen), "recorded_words": len(entries), "record_min": record_min,
                     "max_weights": max_weights, "coverage": coverage, **outcomes}


def write_weights(entries, report: dict, path: Path = DATA / "weights.tsv") -> None:
    header = [
        "# Persian syllable weights: each word, the weight strings (u short, - long) its",
        "# occurrences admitted under their poem's metre label, and the evidence.",
        "# pinned: one string fits ~every occurrence. underdetermined: several do (empty",
        "# when too many to prune). variable: a small set covers them. conflict: none does;",
        "# kept with counts and never used. agreement: share the row's weights explain.",
        "# supervision: derived from gold metre labels; results using it are supervised",
        f"# snapshot: {SNAPSHOT}",
        "# split: development only (aruz_eval.held_out is false)",
        f"# derived: {date.today().isoformat()}",
        *(f"# {k}: {v}" for k, v in report.items()),
        "# word\tstatus\tweights\toccurrences\tagreement\tsupport",
    ]
    rows = ["\t".join((e.word, e.status, "|".join(e.weights), str(e.occurrences), f"{e.agreement:.4f}",
                       "|".join(f"{w}:{c}" for w, c in e.support))) for e in entries]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def evaluate(poems, scanner: Scanner, *, split: str = "held_out", limit: int = 2000,
             max_lines: int = MAX_LINES, seed: int = 7) -> dict:
    """Coverage, ambiguity and accuracy, per line and per poem, kept apart."""
    pool = [p for p in poems if p.held_out == (split == "held_out")]
    random.Random(seed).shuffle(pool)
    pool = pool[:limit]
    alias = {label: frozenset(ps) for label, ps in scanner.inventory.label_patterns.items()}
    lines, poem, reasons, confusion = Counter(), Counter(), Counter(), Counter()
    per_metre = defaultdict(Counter)
    admitted = cheapest = scanned = ambiguity = identified = gated = 0
    for gold in pool:
        tally, costs, used = Counter(), Counter(), 0
        text = hemistichs(gold.text)
        for line in text[:max_lines]:
            s = scanner.scan(line)
            if not s.metres:
                lines["abstain"] += 1
                reasons[s.abstention] += 1
                continue
            scanned += 1
            used += 1
            gated += s.gated
            ambiguity += len(s.metres)
            admitted += gold.metre in dict(s.candidates)
            cheapest += gold.metre in s.metres
            for metre, cost in votes(s):
                tally[metre] += 1
                costs[metre] += cost
            if len(s.metres) == 1:
                lines["exact" if s.metres[0] == gold.metre else "wrong"] += 1
            else:
                lines["ambiguous_" + ("including" if gold.metre in s.metres else "excluding") + "_gold"] += 1
        if not used:
            poem["abstain"] += 1
            per_metre[gold.metre]["abstain"] += 1
            continue
        best = winners(tally, costs, len(text))
        if len(best) == 1:
            outcome = "exact" if best[0] == gold.metre else "wrong"
            if outcome == "wrong":
                confusion[gold.metre, best[0]] += 1
        else:
            outcome = "ambiguous_" + ("including" if gold.metre in best else "excluding") + "_gold"
        poem[outcome] += 1
        per_metre[gold.metre][outcome] += 1
        identified += all(m == gold.metre or alias.get(m) == alias.get(gold.metre) for m in best)
    line_total, poem_total = lines.total() or 1, poem.total() or 1
    return {
        "split": split, "snapshot": SNAPSHOT, "labels": len(scanner.inventory),
        "patterns": len(scanner.inventory.patterns), "devices": asdict(scanner.devices),
        "marks": asdict(scanner.marks) if scanner.marks else None,
        "lexicon": scanner.lexicon.summary() if scanner.lexicon else None,
        "poems": len(pool), "lines": line_total,
        "line_rate": {k: round(v / line_total, 4) for k, v in lines.most_common()},
        "gold_admitted_rate": round(admitted / scanned, 4) if scanned else 0.0,
        "gold_among_cheapest_rate": round(cheapest / scanned, 4) if scanned else 0.0,
        "mean_metres_per_line": round(ambiguity / scanned, 2) if scanned else 0.0,
        "lexicon_gated_rate": round(gated / scanned, 4) if scanned else 0.0,
        "abstentions": dict(reasons.most_common()),
        "poem_rate": {k: round(v / poem_total, 4) for k, v in poem.most_common()},
        "poem_metre_identified": round(identified / poem_total, 4),
        "confusions": [{"gold": g, "scanned_as": s, "poems": n} for (g, s), n in confusion.most_common(12)],
        "per_metre": [{"metre": m, "poems": c.total(), "exact": round(c["exact"] / c.total(), 4)}
                      for m, c in sorted(per_metre.items(), key=lambda kv: -kv[1].total())[:20]],
    }


SUPERHEAVY = "S"  # CV̄C or CVCC, counted – ⏑ or one –
LONG_VOWELS, SHORT_VOWELS, LENGTHENED = set("âiu"), set("aeoĭŭ"), set("ēōā")


def weights_of(pronunciation: str) -> tuple[str, ...]:
    """Weights of a syllabified reading: syllables split by ".", words by spaces;
    a e o short, â i u long, ē ō ā lengthened by the metre, ĭ ŭ shortened; one
    character per consonant."""
    out = []
    for syllable in pronunciation.replace(" ", ".").split("."):
        if not syllable:
            continue
        nucleus = [k for k, c in enumerate(syllable) if c in LONG_VOWELS | SHORT_VOWELS | LENGTHENED]
        if len(nucleus) != 1:
            raise ValueError(f"{pronunciation!r}: syllable {syllable!r} needs one vowel")
        v, coda = syllable[nucleus[0]], len(syllable) - nucleus[0] - 1
        if v in LENGTHENED:
            out.append(LONG)
        elif (v in LONG_VOWELS and coda) or coda >= 2:
            out.append(SUPERHEAVY)
        else:
            out.append(LONG if v in LONG_VOWELS or coda else SHORT)
    return tuple(out)


def realizes(weights, pattern: str) -> bool:
    """Do these weights fill the pattern, with the scanner's position licences?"""
    def go(i, j):
        if i == len(weights):
            return j == len(pattern)
        if i == len(weights) - 1:  # the last syllable fills the last position whatever it weighs
            return j == len(pattern) - 1
        options = {"S": ("-u", "-", "uu"), "-": ("-", "uu")}.get(weights[i], (weights[i],))
        if j == 0 and weights[i] != "u":
            options += ("u", "uu")
        return any(pattern.startswith(o, j) and go(i + 1, j + len(o)) for o in options)
    return go(0, 0)


def readings(skel, patterns, devices: Devices, caesuras=None, cap: int = 50_000):
    """Every weight sequence of ``skel`` realizing one of ``patterns``; None past ``cap``."""
    trie = Trie(sorted(set(patterns)), caesuras if devices.caesura else None)
    n, memo, total = len(skel), {}, 0

    class TooMany(Exception):
        pass

    def tokens(weights, dev):
        return (SUPERHEAVY,) if weights == LONG + SHORT or dev & OVERLONG else tuple(weights)

    def go(i, node):
        nonlocal total
        if (i, node) in memo:
            return memo[i, node]
        starts = [(node, ())]
        if _glide(skel, i, devices):
            starts += [(landed, (w,)) for w, _ in _glide_weights(devices) if (landed := trie.walk(node, w)) is not None]
        if trie.caesura and i and skel.word[i] != skel.word[i - 1]:
            starts += [(trie.children[s][CAESURA], prefix) for s, prefix in starts if CAESURA in trie.children[s]]
        out = set()
        for start, prefix in starts:
            for weights, nxt, dev in _moves(skel, i, devices):
                head = prefix + tokens(weights, dev)
                for realized, licence in _pattern_slots(weights, i, devices):
                    if licence and _slot_blocked(licence, trie.depth[start], devices):
                        continue
                    landed = trie.walk(start, realized)
                    if nxt == n:
                        if (landed is not None and trie.terminal[landed] is not None) or any(
                            trie.terminal[c] is not None and c != landed for c in trie.children[start].values()
                        ):
                            out.add(head)
                        continue
                    if landed is not None:
                        out.update(head + tail for tail in go(nxt, landed))
                    if trie.caesura and skel.word[nxt] != skel.word[nxt - 1]:
                        spilled = trie.spill(start, realized)
                        if spilled is not None:
                            out.update(head + tail for tail in go(nxt, spilled))
                total += 1
                if len(out) > cap or total > 50 * cap:
                    raise TooMany
        memo[i, node] = frozenset(out)
        return memo[i, node]

    try:
        return go(0, 0)
    except TooMany:
        return None


def load_readings(path: Path) -> list[dict]:
    keys = ("id", "doc_id", "metre", "line", "pronunciation", "provenance", "note")
    return [dict(zip(keys, row.split("\t"))) for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip() and not row.startswith("#")]


def score_readings(rows, scanner: Scanner) -> dict:
    """How many hand syllabifications the scanner still reaches under the gold
    metre, and how many readings compete with them."""
    counted, admitted, sizes = 0, 0, []
    rows = [r for r in rows if not r["provenance"].startswith("excluded:")]
    for r in rows:
        found = readings(prepare(r["line"], marks=scanner.marks), scanner.inventory.label_patterns.get(r["metre"], []),
                         scanner.devices, scanner.inventory.caesuras)
        if found is not None:
            counted += 1
            admitted += weights_of(r["pronunciation"]) in found
            sizes.append(len(found))
    sizes.sort()
    mid = len(sizes) // 2
    median = 0.0 if not sizes else float(sizes[mid]) if len(sizes) % 2 else (sizes[mid - 1] + sizes[mid]) / 2
    return {"lines": len(rows), "too_many_to_count": len(rows) - counted, "reading_admitted": admitted,
            "median_readings": median}


def evaluate_couplets(path: Path, scanner: Scanner, *, held_out_only: bool = False) -> dict:
    """Couplets labelled with a metre's feet (Ganjoor-Rhythm-Bench's ``first``,
    ``second``, ``rhythm``, ``poem_id``): is the couplet's metre exactly the label?
    ``held_out_only`` keeps the couplets of held-out poems, for a scanner whose
    lexicon saw the others."""
    rows, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["poem_id"], r["couplet_index"])
            if key not in seen and (not held_out_only or held_out(str(r["poem_id"]))):
                seen.add(key)
                rows.append(r)
    feet = {label: tidy(label).split(" (")[0] for label in scanner.inventory.label_patterns}
    outcome, per_metre = Counter(), defaultdict(Counter)
    for r in rows:
        gold = tidy(r["rhythm"])
        result = scan_poem(r["first"] + "\n" + r["second"], scanner)
        got = {feet.get(m, m) for m in result.metres}
        o = "abstain" if not got else "exact" if got == {gold} else \
            "ambiguous_including_gold" if gold in got else "wrong"
        outcome[o] += 1
        per_metre[gold][o] += 1
    n = outcome.total() or 1
    return {"couplets": outcome.total(), "rate": {k: round(v / n, 4) for k, v in outcome.most_common()},
            "metres": len(per_metre),
            "known_metres": sum(any(f == m for f in feet.values()) for m in per_metre),
            "per_metre": {m: round(c["exact"] / c.total(), 3) for m, c in sorted(per_metre.items(), key=lambda kv: -kv[1].total())}}
