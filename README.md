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

### `vowels`: the vowels editors write

Persian leaves short vowels unwritten, but Ganjoor's editors mark many of
them. The lexicon groups every marked spelling of a consonant skeleton into
readings: two spellings are two readings only when they put different vowels
on the same letter. Marks on the last letter (the ezāfe, an Arabic case ending,
a pausal sukun) are split off as syntax. A skeleton nobody marked is absent,
which means unknown, not unambiguous.

```python
from ghazal_nlp import vowels
[(r.form, r.count) for r in vowels.lexicon().lookup("گل").readings]  # گِل 610, گُل 371, گَل 5
```

From the whole Ganjoor snapshot: 30,259 skeletons have a reading, 27,702 of
them one and 2,557 more than one (1,006 with two readings attested at least
twice each). Over Ganjoor's 20.1M running tokens, 46.5% have an ambiguous
skeleton, 35.2% a settled one and 18.3% one nobody marked. Counts are of marked
spellings: editors mark where a word is hard, so they say how a word is read
when someone bothered to write it down, not how often each reading occurs.
`data/vowels-review.tsv` lists every skeleton with more than one reading, by
reason. Reproduce: `ghazal-nlp vowels-derive poems.jsonl`.

### `ezafe`: the ezāfe after a silent ه

`sites` finds every word ending in a silent ه followed by another word, records
how the junction is written (هٔ، ۀ، ه‌ی، هی، ه ی، or nothing), and removes only
that mark. Two classifiers read only the neutralised text: `written` (is a
written mark an ezāfe, or an old spelling of ‌ای?) and `junction` (does an
unmarked junction carry one?).

```python
from ghazal_nlp import ezafe
for site in ezafe.sites({"text": "نامه را نوشتم و خانه زیبا بود"}):
    print(site["head"], ezafe.junction(site))
```

Measured against Ali Julaee Rad's rulings on development sites from Wikipedia,
Wikisource, books and Ganjoor (`data/ezafe-*-dev.jsonl`). The blind ruling was
made with the mark hidden; the revised one after it was shown and the case
discussed.

| | sites | blind | revised |
|---|---:|---:|---:|
| `written`: is the written mark an ezāfe | 124 | 118 / 123 | 122 / 124 |
| `junction`: is there an ezāfe | 176 | 85.6% | 84.0% |

The development rulings are the ones the rules were written against, so these
are development numbers. `junction` overcalls the ezāfe on unmarked text (24
of its 28 revised errors), because without a noun and adjective lexicon any
word it cannot place counts as content. Reproduce: `ghazal-nlp ezafe-eval`.

## Data

The code is MIT. The data in `ghazal_nlp/data` is CC BY 4.0, and each file's
header says what it was derived from. Ganjoor and the treebanks are read for
measurement and are not shipped.

## References

What the tools are built from and measured on:

- Ganjoor: [ganjoor.net](https://ganjoor.net), its [API](https://api.ganjoor.net) and [source](https://github.com/ganjoor). The poems, the editors' marks and the metre labels behind `aruz`, `vowels` and `ezafe`.
- Finn Thiesen, *A Manual of Classical Persian Prosody, with Chapters on Urdu, Karakhanidic and Ottoman Prosody*. Wiesbaden: Harrassowitz, 1982. ISBN 3-447-02104-7. The source of `readings-thiesen.tsv`, and of the است contraction, the robāʿī family and the دوری caesura in `aruz`.
- امید طبیب‌زاده (به کوشش)، *وزن شعر فارسی از دیروز تا امروز* (مجموعهٔ سخنرانی‌های نخستین هم‌اندیشی وزن شعر فارسی و اشعار ایرانی در انجمن زبان‌شناسی ایران، ۹ و ۱۰ اردیبهشت ۱۳۸۹). تهران: فرهنگستان زبان و ادب فارسی، نشر آثار، ۱۳۹۰. ISBN 978-600-6143-12-5. Najafi's «دایره‌ای با ۱۸۰ وزن» (one metre is what can co-occur in one poem; the دوری halves), Shamisa on ابدال in the first foot, and Samiʿi Gilani's table of Saʿdī's metres, used to check Ganjoor's labels.
- John Mace, *Persian Grammar: For Reference and Revision*. London: RoutledgeCurzon, 2003. Appendix I, the irregular present stems behind `verb-stems.tsv`.
- فرهنگستان زبان و ادب فارسی، *دستور خطّ فارسی*، ویراست جدید، ۱۴۰۱ ([PDF](https://apll.ir/wp-content/uploads/2023/07/Dastour-e-Khat-17.04.1402.pdf)). The orthography rules, and the silent-ه spellings (pp. 46–47) and stem changes after ب، ن and می (pp. 38–39).
- [لغت‌نامهٔ دهخدا](https://dehkhoda.ut.ac.ir), consulted on doubtful verbs (اندودن، فراختن).
- [UD Persian Seraji](https://github.com/UniversalDependencies/UD_Persian-Seraji): Mojgan Seraji, Filip Ginter and Joakim Nivre, [Universal Dependencies for Persian](https://aclanthology.org/L16-1374/), LREC 2016.
- [UD Persian PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT): Mohammad Sadegh Rasooli et al., [The Persian Dependency Treebank Made Universal](https://arxiv.org/abs/2009.10205), 2020.
- [iPerUDT](https://github.com/royakabiri/iPerUDT): Roya Kabiri, Simin Karimi and Mihai Surdeanu, [Informal Persian Universal Dependency Treebank](https://aclanthology.org/2022.lrec-1.766/), LREC 2022.
- Charles Elkan and Keith Noto, [Learning Classifiers from Only Positive and Unlabeled Data](https://cseweb.ucsd.edu/~elkan/posonly.pdf), KDD 2008. The learned ezāfe classifier on the `dev` branch.

Other Persian tools and data we found useful:

- [Hazm](https://github.com/roshan-research/hazm), [Parsivar](https://github.com/ICTRC/Parsivar), [DadmaTools](https://github.com/Dadmatech/DadmaTools) and [Stanza](https://github.com/stanfordnlp/stanza): normalisation, tokenisation, tagging and parsing pipelines.
- [Perstem](https://github.com/jonsafari/perstem) and [Persian-Morphology](https://github.com/omidrohanian/Persian-Morphology): a stemmer and a two-level morphology in foma.
- [PARSEME shared task data](https://gitlab.com/parseme/sharedtask-data), including Persian multiword expressions.
- [ParsMap](https://huggingface.co/datasets/sbunlp/ParsMap): formal and informal Persian pairs.
- [farhangestan](https://github.com/5j9/farhangestan): the Academy's approved terms as data.
