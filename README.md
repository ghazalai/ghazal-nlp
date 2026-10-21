# ghazal-nlp

Small deterministic tools for Persian. No models, no network, just the Python
standard library. Every answer says which rule produced it, and when the rules
can't decide, the tool abstains instead of guessing.

```sh
pip install git+https://github.com/alijulaeerad/ghazal-nlp
ghazal-nlp scan "الا یا ایها الساقی ادر کأسا و ناولها"
ghazal-nlp verb "نمی‌گفتند برمی‌دارد"
ghazal-nlp conjugate گفتن --tense past --aspect imperfective --person 1 --number plur
ghazal-nlp check "آنها کتاب ها را میخوانند"
```

## Tools

### `aruz`: classical metre

Persian doesn't write short vowels, the ezāfe or gemination, so one written
hemistich can be read several ways. The scanner tries those readings against
146 weight patterns (142 metres). For each line it returns the cheapest
metres, every other metre that fits along with the licences it needs (hamze
elision, و read short, ابدال, an overlong syllable read long, …), or an
abstention with a reason. Harakat, shadda, the ezāfe hamza and ZWNJ written by
the editor are used as constraints. A poem gets the metre most of its lines
fit.

```python
from ghazal_nlp import aruz
s = aruz.scanner().scan("الا یا ایها الساقی ادر کأسا و ناولها")
s.metres, s.weights, s.devices, s.candidates
aruz.scan_poem("بشنو این نی چون شکایت می‌کند\nاز جدایی‌ها حکایت می‌کند").metres
```

Tested on 2,000 held-out Ganjoor poems (32,084 hemistichs) against Ganjoor's
metre labels. The held-out poems are a fixed fifth, chosen by poem id.

| | rules only | with the weight lexicon |
|---|---:|---:|
| gold metre among the answers | 99.2% | 95.7% |
| metres per hemistich | 3.07 | 1.80 |
| poem gets exactly the gold metre | 78.9% | 93.5% |
| abstained | 0.05% | 0.05% |

The weight lexicon (`data/weights.tsv`) is learned from the labels of the
other four fifths, so that column is supervised. The rules-only column isn't.

Thiesen's *Manual of Classical Persian Prosody* scans 513 hemistichs; the
scanner reaches his exact weights for 494 of them.

```sh
ghazal-nlp ganjoor <snapshot> > poems.jsonl
ghazal-nlp aruz-eval poems.jsonl [--no-lexicon] [--stripped]
ghazal-nlp aruz-derive poems.jsonl   # rebuilds metres.tsv and weights.tsv
```

### `verb`: verb analysis and generation

Every reading of a verb token: tense, aspect, mood, person, number, polarity,
participles and perfects. It knows 251 verbs and 94 separable preverb pairs
(بر، در، باز، وا، فرو، سر، فرا), and abstains on a stem it doesn't know.
`constructions` reads two-word tenses (رفته بود، خواهد رفت، گفته شد), and
`conjugate` builds any form.

```python
from ghazal_nlp import verb
[r.features() for r in verb.analyzer().analyze("نمی‌کرده‌ایم").readings]
verb.analyzer().conjugate("برداشتن", mood="imperative", person=2)  # بردار
```

Tested on every VERB and AUX word in the UD Persian treebanks:

| | Seraji | PerDT | iPerUDT (informal) |
|---|---:|---:|---:|
| verbs | 17,680 | 72,817 | 8,911 |
| analysed | 82.2% | 89.0% | 55.4% |
| features agree, when analysed | 99.3% | 98.8% | 97.8% |
| readings per analysed verb | 1.14 | 1.14 | 1.13 |
| می‌ and نمی‌ forms analysed | 99.2% | 98.9% | 70.3% |

Most of what's left is the copula است and colloquial forms (میشه، داره). The
treebanks don't agree with each other on lemmas and some features, so lemmas
and features are scored separately; `verb_eval.py` has the exact rule.

```sh
ghazal-nlp verb-eval <treebank>/*.conllu
```

### `vowels`: the vowels editors write

Ganjoor's editors mark short vowels on the words they think need them. This
lexicon groups every marked spelling of a word into readings: two spellings
are different readings only if they put different vowels on the same letter.
Marks on the last letter (ezāfe, Arabic case endings) are split off. A word
nobody marked isn't in the lexicon, so missing means unknown, not unambiguous.

```python
from ghazal_nlp import vowels
[(r.form, r.count) for r in vowels.lexicon().lookup("گل").readings]  # گِل 610, گُل 371, گَل 5
```

Over all of Ganjoor: 30,259 words have a reading, 2,557 of them more than one.
Of 20.1M running words, 46.5% have more than one reading, 35.2% have one, and
18.3% were never marked. The counts are of marked spellings, so they show how
a word is read where an editor thought it was worth marking, not how often
each reading occurs. `data/vowels-review.tsv` lists the words with more than
one reading.

```sh
ghazal-nlp vowels-derive poems.jsonl
```

### `ezafe`: the ezāfe after a silent ه

`sites` finds every word ending in a silent ه that's followed by another word,
records how the junction is written (هٔ، ۀ، ه‌ی، هی، ه ی or nothing), and
removes just that mark. Two classifiers then look at the unmarked text:
`written` decides whether a written mark is an ezāfe or an old spelling of ‌ای,
and `junction` decides whether an unmarked junction has an ezāfe.

```python
from ghazal_nlp import ezafe
for site in ezafe.sites({"text": "نامه را نوشتم و خانه زیبا بود"}):
    print(site["head"], ezafe.junction(site))
```

Tested by me on sites from Wikipedia, Wikisource, books and Ganjoor
(`data/ezafe-*-dev.jsonl`):

| | sites | agree |
|---|---:|---:|
| `written` | 124 | 122 (98%) |
| `junction` | 175 | 147 (84%) |

The rules were written while looking at these sites, so treat these as
development numbers. `junction` mostly errs by finding an ezāfe that isn't
there, since without a noun and adjective lexicon it treats any word it can't
place as a noun.

```sh
ghazal-nlp ezafe-eval
```

### `orthography`: spelling rules

Two styles. `academy` is the Persian Academy's دستور خطّ فارسی, each rule
citing its page. `ghazal` is my house style: I treat the Academy as evidence,
not authority. It adopts the Academy rules careful writers actually follow,
and where they don't, it goes with what careful contemporary writing does.

Every rule is a TOML record (`rules/`) with the registers it applies to,
correct examples it must not flag and wrong ones it must. A rule that fails
its own examples doesn't load. When a rule can't decide, it abstains and
says why (a ZWNJ before ای is only suggested once the word's final ه is known
to be silent). In `ghazal`, a finding is either a `violation` (something no
careful writer does: Arabic ي, a full space before ها) or `house` (both
spellings are fine, here's the house one).

```python
from ghazal_nlp import orthography
for f in orthography.rulebase("ghazal").check("آنها کتاب ها را میخوانند").findings:
    print(f.rule_id, f.severity, f.text, "→", f.suggestion)
orthography.is_silent_he("خانه"), orthography.is_silent_he("راه")  # True, False
```

Where the two styles differ:

| | `academy` | `ghazal` |
|---|---|---|
| ezāfe after silent ه | خانهٔ / خانۀ; خانه‌ی is wrong | خانه‌ی; خانهٔ and خانۀ are fine |
| plural and comparative | ZWNJ, joined allowed | ZWNJ (house); joined is fine |
| compounds (آنها، این‌طور، هیچ‌کدام) | its printed lists | what careful writers write, pair by pair (`data/ghazal-house-forms.tsv`) |
| tanvin (واقعاً) | required | left as written |
| preferred spellings (اطاق/اتاق) | the Academy's pick | both fine |
| می before a verb | ZWNJ; spaced form only | ZWNJ; spaced or joined, checked by `verb` |
| digits, «», …, Latin , ; ? | not covered | Persian digits, « », …, ، ؛ ؟ |

The silent-ه list (`data/orthography-silent-he.tsv`, 1,253 words) comes from
how each word's own spellings treat the ه (خانه‌ای vs گرهی).

```sh
ghazal-nlp check "آنها کتاب ها را میخوانند"
ghazal-nlp check --style academy "آنها کتاب ها را میخوانند"
ghazal-nlp rules [--style academy]
ghazal-nlp verify
```

### `arabic`: Arabic passages inside Persian

Finds Arabic passages inside Persian text, conservatively. A shared word, a
name or a ة isn't enough, Persian words end a span, and finding nothing means
"don't know", not "no Arabic". A second, exact matcher finds Quran wording,
given a copy of the text (Tanzil format).

```sh
ghazal-nlp arabic "چنانکه گفته‌اند الحمد لله رب العالمین و ..."
```

## Data

The code is MIT. The data in `ghazal_nlp/data` is CC BY 4.0; each file's
header says where it comes from. Ganjoor and the treebanks are only read for
testing and aren't included.

## References

What the tools are built from and tested on:

- Ganjoor: [ganjoor.net](https://ganjoor.net), its [API](https://api.ganjoor.net) and [source](https://github.com/ganjoor). The poems, the editors' marks and the metre labels behind `aruz`, `vowels` and `ezafe`.
- Finn Thiesen, *A Manual of Classical Persian Prosody, with Chapters on Urdu, Karakhanidic and Ottoman Prosody*. Wiesbaden: Harrassowitz, 1982. ISBN 3-447-02104-7. The source of `readings-thiesen.tsv`, and of the است contraction, the robāʿī family and the دوری caesura in `aruz`.
- امید طبیب‌زاده (به کوشش)، *وزن شعر فارسی از دیروز تا امروز* (مجموعهٔ سخنرانی‌های نخستین هم‌اندیشی وزن شعر فارسی و اشعار ایرانی در انجمن زبان‌شناسی ایران، ۹ و ۱۰ اردیبهشت ۱۳۸۹). تهران: فرهنگستان زبان و ادب فارسی، نشر آثار، ۱۳۹۰. ISBN 978-600-6143-12-5. Najafi's «دایره‌ای با ۱۸۰ وزن» (one metre is what can co-occur in one poem; the دوری halves), Shamisa on ابدال in the first foot, and Samiʿi Gilani's table of Saʿdī's metres, used to check Ganjoor's labels.
- John Mace, *Persian Grammar: For Reference and Revision*. London: RoutledgeCurzon, 2003. Appendix I, the irregular present stems behind `verb-stems.tsv`.
- فرهنگستان زبان و ادب فارسی، *دستور خطّ فارسی*، ویراست جدید، ۱۴۰۱ ([PDF](https://apll.ir/wp-content/uploads/2023/07/Dastour-e-Khat-17.04.1402.pdf)). The `academy` rules, the silent-ه spellings (pp. 46–47) and the stem changes after ب، ن and می (pp. 38–39).
- [لغت‌نامهٔ دهخدا](https://dehkhoda.ut.ac.ir), for doubtful verbs (اندودن، فراختن).
- [UD Persian Seraji](https://github.com/UniversalDependencies/UD_Persian-Seraji): Mojgan Seraji, Filip Ginter and Joakim Nivre, [Universal Dependencies for Persian](https://aclanthology.org/L16-1374/), LREC 2016.
- [UD Persian PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT): Mohammad Sadegh Rasooli et al., [The Persian Dependency Treebank Made Universal](https://arxiv.org/abs/2009.10205), 2020.
- [iPerUDT](https://github.com/royakabiri/iPerUDT): Roya Kabiri, Simin Karimi and Mihai Surdeanu, [Informal Persian Universal Dependency Treebank](https://aclanthology.org/2022.lrec-1.766/), LREC 2022.
- Charles Elkan and Keith Noto, [Learning Classifiers from Only Positive and Unlabeled Data](https://cseweb.ucsd.edu/~elkan/posonly.pdf), KDD 2008. The learned ezāfe classifier on the `dev` branch.

- L. P. Elwell-Sutton, *The Persian Metres*. Cambridge University Press, 1976.
- Aso Mahmudi and Hadi Veisi, [Automatic Meter Classification of Kurdish Poems](https://arxiv.org/abs/2102.12109), 2021: the same design as `aruz` (enumerate weight readings, score over the poem), for Kurdish.

Other Persian tools and data I found useful:

- [Hazm](https://github.com/roshan-research/hazm), [Parsivar](https://github.com/ICTRC/Parsivar), [DadmaTools](https://github.com/Dadmatech/DadmaTools) and [Stanza](https://github.com/stanfordnlp/stanza): normalisation, tokenisation, tagging and parsing.
- [Perstem](https://github.com/jonsafari/perstem) and [Persian-Morphology](https://github.com/omidrohanian/Persian-Morphology): a stemmer and a two-level morphology in foma.
- [PARSEME shared task data](https://gitlab.com/parseme/sharedtask-data), including Persian multiword expressions.
- [ParsMap](https://huggingface.co/datasets/sbunlp/ParsMap): formal and informal Persian pairs.
- [farhangestan](https://github.com/5j9/farhangestan): the Academy's approved terms as data.
- [Shekar](https://github.com/amirivojdan/shekar): an Academy-guided normaliser and spell checker, the closest toolkit to this one.
- Pegah Nourian et al., [On the Importance of Ezafe Construction in Persian Parsing](https://aclanthology.org/P15-2144/), ACL 2015, an ezāfe layer on the dependency treebank; and Ehsan Doostmohammadi et al., [Persian Ezafe Recognition Using Transformers and Its Role in Part-Of-Speech Tagging](https://arxiv.org/abs/2009.09474), 2020.
- [GE2PE](https://github.com/Sharif-SLPL/GE2PE) and [HomoRich](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian): Persian G2P with ezāfe and homograph sets.
- [PerLex](https://aclanthology.org/L10-1486/), a morphological lexicon, and [PersPred](https://aclanthology.org/W13-1002/), light-verb constructions.
