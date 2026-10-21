# ghazal-nlp

Persian rules as small deterministic tools. No model, no network, nothing
beyond the Python standard library. Every answer names the rule or licence
behind it, and where the rules cannot decide, the tool says so instead of
guessing.

```sh
pip install git+https://github.com/alijulaeerad/ghazal-nlp
ghazal-nlp scan "الا یا ایها الساقی ادر کأسا و ناولها"
ghazal-nlp verb "نمی‌گفتند برمی‌دارد"
ghazal-nlp conjugate گفتن --tense past --aspect imperfective --person 1 --number plur
```

## Tools

### `aruz`: classical metre

A written hemistich admits several syllabifications, because Persian does not
write short vowels, the ezāfe or gemination. The scanner searches them against
the 146 weight patterns of 142 metres and reports the cheapest metres, every
other metre it admits with its cost in licences (hamze elision, the conjunction
و, ابدال, an overlong read long, …), or an abstention with a reason. It reads
an editor's harakat, shadda, ezāfe hamza and ZWNJ as constraints. A poem's
metre is the one most of its hemistichs admit.

```python
from ghazal_nlp import aruz
s = aruz.scanner().scan("بشنو این نی چون شکایت می‌کند")
s.metres, s.weights, s.devices, s.candidates
aruz.scan_poem(poem_text).metres
```

Measured on 2,000 held-out Ganjoor poems (32,084 hemistichs), against
Ganjoor's own metre labels (snapshot 2026-08-03; a fixed fifth of the poems is
held out by id):

| | rules alone | with the weight lexicon |
|---|---:|---:|
| hemistichs whose gold metre is admitted | 99.2% | 95.7% |
| metres admitted per hemistich | 3.07 | 1.80 |
| poems scanned to exactly the gold metre | 78.9% | 93.5% |
| hemistichs abstained on | 0.05% | 0.05% |

The weight lexicon (`data/weights.tsv`) records the syllable weights each word
took under its poem's metre label, from the development poems only. It is
supervised; the rules-alone column is not.

Of the 513 hemistichs Thiesen scans in *A Manual of Classical Persian Prosody*
(1982), the scanner can produce his exact syllable weights for 494
(`data/readings-thiesen.tsv`).

Reproduce: `ghazal-nlp ganjoor <snapshot> > poems.jsonl`, then
`ghazal-nlp aruz-eval poems.jsonl [--no-lexicon] [--stripped]`.
`ghazal-nlp aruz-derive poems.jsonl` rebuilds `metres.tsv` and `weights.tsv`.

### `verb`: verb analysis and generation

Every reading of a verb token, from the stems of 251 verbs and 94 attested
separable-preverb pairs (بر، در، باز، وا، فرو، سر، فرا): tense, aspect, mood,
person, number, polarity, participles, perfects and the imperfective perfect,
each naming the rules that built it. An unknown stem abstains. `constructions`
reads two-token tenses (رفته بود، خواهد رفت، گفته شد) without changing the
token readings, and `conjugate` generates any cell of the paradigm.

```python
from ghazal_nlp import verb
[r.features() for r in verb.analyzer().analyze("نمی‌کرده‌ایم").readings]
verb.analyzer().conjugate("برداشتن", mood="imperative", person=2)  # بردار
```

Measured on every VERB and AUX word of the UD Persian treebanks:

| | Seraji | PerDT | iPerUDT (informal) |
|---|---:|---:|---:|
| verbs | 17,680 | 72,817 | 8,911 |
| analysed | 82.2% | 89.0% | 55.4% |
| features agree, when analysed | 99.3% | 98.8% | 97.8% |
| readings per analysed verb | 1.14 | 1.14 | 1.13 |
| می‌ and نمی‌ words analysed | 99.2% | 98.9% | 70.3% |

Most unanalysed verbs are the copula است and colloquial forms (میشه، داره).
The treebanks' conventions differ from each other and from the analyser's
(Seraji lemmatises passive شد as کرد; the future auxiliary carries Tense=Fut),
so features and lemmas are scored apart, by the rule stated in `verb_eval.py`.
Reproduce: `ghazal-nlp verb-eval <treebank>/*.conllu`.

## Data

The code is MIT. The data in `ghazal_nlp/data` is CC BY 4.0, and each file's
header says what it was derived from. Ganjoor and the treebanks are read for
measurement and are not shipped.
