# ghazal-nlp

NLP tools for Farsi and its variations, python standard library were possible.

```sh
pip install ghazal-nlp
ghazal-nlp scan "یار با ماست، چه حاجت که زیادت طلبیم"
ghazal-nlp verb "نمی‌گفتند برمی‌دارد"
ghazal-nlp conjugate گفتن --tense past --aspect imperfective --person 1 --number plur
ghazal-nlp check "آنها کتاب ها را میخوانند"
```

## Tools

### `aruz`: classical metre

Persian doesn't write short vowels, the ezāfe or gemination, so one written
hemistich can be read several ways. The scanner tries those readings against
76 weight patterns (65 metres, the ones at least five Ganjoor poems use). For each line it returns the cheapest
metres, every other metre that fits along with the licences it needs (hamze
elision, و read short, ابدال, an overlong syllable read long, …), or an
abstention with a reason. Harakat, shadda, the ezāfe hamza and ZWNJ written by
the editor are used as constraints. A poem gets the metre most of its lines
fit; among equals, the one with fewer licences and more poems behind it.

```python
from ghazal_nlp import aruz
s = aruz.scanner().scan("یار با ماست، چه حاجت که زیادت طلبیم")
s.metres, s.weights, s.devices, s.candidates
aruz.scanner().readings("یار با ماست، چه حاجت که زیادت طلبیم")[0].syllables  # which letters fill which slot
aruz.scan_poem("بشنو این نی چون شکایت می‌کند\nاز جدایی‌ها حکایت می‌کند").metres
```

Tested on 2,000 held-out Ganjoor poems (32,084 hemistichs) against Ganjoor's
metre labels. The held-out poems are a fixed fifth, chosen by poem id.

| | rules only | with the weight lexicon |
|---|---:|---:|
| gold metre among the answers | 99.1% | 95.3% |
| gold metre among the cheapest | 68.4% | 77.0% |
| metres per hemistich | 2.19 | 1.44 |
| poem gets exactly the gold metre | 92.5% | 95.9% |
| abstained | 0.08% | 0.08% |

The weight lexicon (`data/weights.tsv`) is learned from the labels of the
other four fifths, so that column is supervised. The rules-only column isn't.
With the editors' marks removed (`--stripped`), a poem gets exactly the gold
metre 93.1% of the time with rules only and 96.0% with the lexicon.
Holding out whole poets instead of poems (`--by poet`, 2,000 poems by poets
the lexicon never saw), that's 92.4% and 95.5%.

A few lines are enough for most poems. On 2,000 other held-out poems, reading only
their opening lines, the exact metre comes out:

| hemistichs given | 1 | 2 | 3 | 4 | 8 | whole poem |
|---|---:|---:|---:|---:|---:|---:|
| exact metre | 82.8% | 89.1% | 93.5% | 93.6% | 95.2% | 96.1% |
| wrong | 17.2% | 10.9% | 6.5% | 6.4% | 4.8% | 3.9% |

On single couplets with no poem around them it's much weaker. On the 12,713
couplets of [Ganjoor-Rhythm-Bench](https://huggingface.co/datasets/farbodbij/Ganjoor-Rhythm-Bench)
(40 metres), the rules alone give exactly the gold metre for 50.7%. With the
lexicon, on the 2,553 couplets from held-out poems, that's 70.0%.

On those couplets, next to [an XLM-R metre classifier](https://huggingface.co/katomyomachia/xlm-roberta-persian-meter)
(35 metres, trained on Ganjoor hemistichs; its card reports 98% on a tenth of
its own data):

| exact metre per couplet | couplets | XLM-R | `aruz` with the lexicon | always the commonest metre |
|---|---:|---:|---:|---:|
| all held-out | 2,553 | 61.9% | 70.0% | 4.0% |
| metre the model knows | 2,204 | 71.6% | 73.6% | 4.6% |
| … and no hemistich in its training data | 2,041 | 70.9% | 73.8% | 4.3% |

On lone couplets in its own metres `aruz` is about 2 points ahead, and across all
metres about 8, since 13.7% of the couplets are in metres the classifier can't
answer. The GPT2-fa classifier reported at 90.4%
([kasraSMD/Persian-Poetry-Rhythm-Detection](https://github.com/kasraSMD/Persian-Poetry-Rhythm-Detection))
isn't here: its weights are behind a Google sign-in.

Thiesen's *Manual of Classical Persian Prosody* scans 513 hemistichs; the
scanner reaches his exact weights for 506 of them.

Where a line leaves the metre, `nearest` reads it in a given pattern with the
fewest breaks and marks each one on its syllable. `sakta` marks where a line
fits but the ear may stumble, after Shamisa's chapter «سکتهٔ عروضی»: an ابدال
before the last syllable, heavy where it crosses two feet or sits next to
another licence.

```python
s = aruz.scanner()
s.nearest("کز نیستان تا مرا از بن ببریده‌اند", "-u---u---u-")  # each Syllable has .fault
aruz.sakta(s.readings("ز قدر و رتبت، سر بر ستارگان ساید")[0])  # [(syllable index, "light" or "heavy"), …]
```

On 800 lines from held-out poems, each with one word dropped, added or swapped,
and the same 800 untouched, scored against the median licences of their poem's
first eight hemistichs:

| | broken | untouched |
|---|---:|---:|
| no reading in the poem's metre | 40.6% | 2.0% |
| no reading, or at least 3 licences above the median | 58.0% | 5.2% |

So about four in ten broken lines still scan, because the licences are
generous. Where `nearest` marked a break, it was on the changed word 45.5% of
the time and within one word 79.8%. On the 11 lines of Shamisa's chapter tried,
`sakta` lands on the word he names in 9. It marks 10.8% of Hafez's lines (4.0%
heavy) and 18.0% of Masud Sa'd's (10.8% heavy). The direction agrees with
Shamisa, but he finds only a few in Hafez, so it over-marks.

```sh
ghazal-nlp ganjoor <snapshot> > poems.jsonl
ghazal-nlp aruz-eval poems.jsonl [--no-lexicon] [--stripped]
ghazal-nlp aruz-derive poems.jsonl --out ghazal_nlp/data
ghazal-nlp aruz-derive poems.jsonl --by poet --out poet/
ghazal-nlp aruz-eval poems.jsonl --by poet --weights poet/weights.tsv
ghazal-nlp aruz-baselines benchmark.jsonl xlmr.jsonl
```

### `rhyme`: qāfiya and radif

Give it a poem, one hemistich per line, and its form (ghazal, qasida, qeta,
masnavi, robai, dobeyti). It finds the radif, the words every rhyming
hemistich ends with, and the qāfiya, the letters before the radif that they
share. It also gives each line's rhyme word, flags any line that misses the
rhyme, and lists rhyme words used twice. The even hemistichs decide the rhyme,
and the opening one is checked against them, since a robāʿī's first line
doesn't have to rhyme. A masnavi gets one rhyme per couplet. Matching is on
letters only: harakat, ZWNJ and punctuation don't count, and an ezāfe هٔ counts
as هی.

```python
from ghazal_nlp import rhyme
r, = rhyme.scan(["دلم جز مهر مه‌رویان طریقی بر نمی‌گیرد", "ز هر در می‌دهم پندش ولیکن در نمی‌گیرد",
                 "خدا را ای نصیحتگو حدیث ساغر و می گو", "که نقشی در خیال ما از این خوشتر نمی‌گیرد"], "ghazal")
r.qafiya, r.radif, r.words  # ر, نمی‌گیرد, (بر, در, خوشتر)
```

Tested against the rhyme letters Ganjoor gives on the held-out fifth of its
poems:

| | poems | same rhyme letters as Ganjoor | abstained |
|---|---:|---:|---:|
| ghazal | 12,293 | 97.5% | 0.8% |
| qasida | 1,563 | 92.5% | 2.1% |
| qeta | 1,621 | 95.1% | 2.6% |
| robāʿī | 4,750 | 98.8% | 0.1% |
| dobeyti | 79 | 96.2% | 0% |

Ganjoor has no rhyme letters for a masnavi; 98.5% of the 96,707 held-out
masnavi couplets rhyme. Most of the disagreements are poems where one line
misses the rhyme: Ganjoor keeps only what every line shares, while I keep what
nearly all of them share and flag the odd line out.

```sh
ghazal-nlp rhyme --form ghazal < poem.txt
ghazal-nlp ganjoor <snapshot> > poems.jsonl
ghazal-nlp rhyme-eval poems.jsonl [--misses 20]
```

### `verb`: verb analysis and generation

Every reading of a verb token: tense, aspect, mood, person, number, polarity,
participles and perfects. It knows 251 verbs and 94 separable preverb pairs
(بر، در، باز، وا، فرو، سر، فرا), the copula, and spoken forms (میشه، کنه،
اومدم، نمیدونم). It abstains on a stem it doesn't know.
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
| analysed | 97.5% | 97.0% | 89.7% |
| features agree, when analysed | 99.5% | 98.9% | 97.5% |
| features and mood agree, where mood is marked | 98.4% | 90.4% | 93.2% |
| readings per analysed verb | 1.30 | 1.35 | 1.24 |
| می‌ and نمی‌ forms analysed | 99.7% | 98.9% | 99.3% |

Most of what's left are pieces the treebanks split off a word (ست، ند، کن)
and, in iPerUDT, a spoken ه written apart. None of the treebanks marks aspect.
They don't agree with each other on lemmas and some features, so lemmas and
features are scored separately; `verb_eval.py` has the exact rule.

`conjugate` against [UniMorph's Persian paradigms](https://github.com/unimorph/fas),
on the cells it builds (present and past imperfective, simple past, present
subjunctive, imperative). UniMorph writes no ZWNJ, so neither side has one.

| | simple verbs | compound verbs |
|---|---:|---:|
| cells | 1,794 | 5,304 |
| built | 85.5% | 99.0% |
| same form, when built | 95.4% | 98.3% |
| every verb regular (baseline) | 76.6% | 50.0% |

Of the 162 cells that differ, 160 are UniMorph's own garbles (ببرخیز، میبه کار
برند، بافکن) and 2 are بگشا where UniMorph has بگشای. Eleven verbs aren't in
my stem list, three of them typos in UniMorph (خندسن، چیندن، رختن).

```sh
ghazal-nlp verb-eval <treebank>/*.conllu
ghazal-nlp conjugate-eval <unimorph>/fas
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

Against full pronunciations of modern prose: [SentenceBench](https://huggingface.co/datasets/MahtaFetrat/SentenceBench)
(400 sentences) and [HomoRich](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian)'s
test split (127,850 sentences on 285 homographs). A reading fits when the
pronunciation has the vowels it marks.

| | SentenceBench | HomoRich |
|---|---:|---:|
| words with one reading: it fits | 80.6% (618) | 82.2% (248,011) |
| words with several: the commonest fits | 53.0% (1,014) | 53.9% (433,153) |
| … one of them fits | 91.2% | 92.5% |
| … a random one fits | 46.1% | 47.6% |
| the homograph itself: the commonest fits | 42.7% (164) | 38.2% (65,955) |
| … a random one fits | 35.4% | 39.3% |

The readings are Ganjoor's, so a miss is often verse against prose (حَرَکت,
نَمی, خَیلی). On the homographs the commonest reading is no better than a
random one. The two sets aren't independent: 71 of SentenceBench's 88 Common
Voice sentences and at least 45 of its 100 ManaTTS ones are in HomoRich word for
word.

```sh
ghazal-nlp vowels-derive poems.jsonl --out ghazal_nlp/data
ghazal-nlp g2p-eval SentenceBench.csv
```

### `ezafe`: the ezāfe after a silent ه

`junctions` finds every word ending in a silent ه that's followed by another word,
records how the junction is written (هٔ، ۀ، ه‌ی، هی، ه ی or nothing), and
removes just that mark. Two classifiers then look at the unmarked text:
`written` decides whether a written mark is an ezāfe or an old spelling of ‌ای,
and `junction` decides whether an unmarked junction has an ezāfe.

```python
from ghazal_nlp import ezafe
for j in ezafe.junctions("نامه را نوشتم و خانه زیبا بود"):
    print(j.head, j.mark, j.label, j.reason)
```

`junction` has two profiles. `prose`, the default, is the one above. `verse`
keeps the rules for where there's no ezāfe, then lets the metre decide: it scans
the line once with the junction read as an ezāfe and once without, and takes
whichever needs fewer licences. Give it the poem's metre when you know it;
otherwise it takes the metre most of the text's hemistichs scan cheapest in, if
there are two or more and 40% of them agree. With no metre or a tie it says
none, so give it the whole poem, not one line.

```python
ezafe.junctions("فرستاده شاه را پیش خواند", profile="verse",
                metre="فعولن فعولن فعولن فعل (متقارب مثمن محذوف یا وزن شاهنامه)")
```

Tested by me on sites from Wikipedia, Wikisource and Ganjoor, and against the
pronunciations in SentenceBench and HomoRich:

| | sites | agree | always the commoner label |
|---|---:|---:|---:|
| `written`, dev | 102 | 100 (98%) | 88.2% |
| `written`, held out | 78 | 68 (87%) | 87.2% |
| `written`, SentenceBench | 14 | 13 (93%) | 100% |
| `written`, HomoRich | 2,724 | 98.4% | 98.8% |
| `junction`, dev | 128 | 111 (87%) | 61.7% |
| `junction`, lab batch 1 | 43 | 37 (86%) | 86.0% |
| `junction`, SentenceBench | 132 | 104 (79%) | 76.5% |
| `junction`, HomoRich | 58,962 | 81.4% | 73.6% |

The dev sites (`data/ezafe-*-dev.jsonl`) were labelled while the rules were
written. Batch 1 (`data/ezafe-lab-1.jsonl`) was held out, but I fixed
`junction` after looking at it (participles before شده، heads like همیشه that
take none, reduplication), so it counts as dev for `junction`. A second batch
is held out for that. `written` doesn't beat always saying ezāfe on held-out
text. `junction` mostly errs by finding an ezāfe that isn't there, since
without a noun and adjective lexicon it treats any word it can't place as a
noun. HomoRich's ezāfe marks were added automatically (a POS tagger and a
dictionary, per its paper), and about 40% of its sentences GPT-4o wrote, so its
rows measure agreement with that tagging, not with people.

```sh
ghazal-nlp ezafe --profile verse --metre "<Ganjoor's label>" < poem.txt
ghazal-nlp ezafe-eval
ghazal-nlp g2p-eval test_set.csv
```

### `ezafe.heh`: the ezāfe written as ه

Informal writing often makes the ezāfe visible with ه: دیواره اتاقه من for
دیوارِ اتاقِ من. `heh` takes every word ending in ه before another word, leaves
out verbs and Arabic, and says whether the ه is that ezāfe (`wrong`, with the
pair rewritten with a kasra) or belongs there (`right`: خونه، خوبه، پسره رفت).

It's 700 boosted trees over corpus counts of the two words (how edited text
spells the word, how often the pair is written bare or with a kasra, what
follows each) and the words around. They trained on kasras rewritten as ه and
on 1,162 sites Claude labelled, and are calibrated to the 2.5% of sites where
the ه is an ezāfe, so they flag a site only when sure.

Tested on 297 sites I labelled from tweets and YouTube comments, weighted to
how often each kind occurs (95% intervals):

| precision | recall | average precision |
|---:|---:|---:|
| 0.86 [0.71, 0.98] | 0.24 [0.13, 0.44] | 0.46 [0.29, 0.75] |

Logistic regression on the same counts did as well on this test; on Claude's
labels the trees are ahead (average precision 0.82 against 0.75). A character
BiGRU and a pretrained character transformer did worse on Claude's labels and
no better on mine.

```sh
ghazal-nlp ezafe-heh "اسمه من علیه"
```

### The Tajik Shāhnāme as gold

Tajik Cyrillic writes the short vowels and the ezāfe. I aligned a Tajik copy of
the Shāhnāme to Ganjoor's, hemistich by hemistich, and put its vowels on the
Persian letters. `data/shahnameh-tajik-test.jsonl` holds 3,000 hemistichs from
the held-out fifth of the poems, each word with its vocalized form and whether
an ezāfe follows; only words the alignment accounted for letter by letter are
kept.

| | words / sites | tool | baseline |
|---|---:|---:|---:|
| `vowels`, one reading: it fits | 5,499 | 87.6% | |
| `vowels`, several: the commonest fits | 6,689 | 59.4% | a random one 46.1% |
| `vowels`, several: one of them fits | 6,689 | 95.5% | |
| `ezafe.junction`, prose | 561 | 54.7% | always none 90.6% |
| `ezafe.junction`, verse, the metre given | 561 | 96.4% | always none 90.6% |
| `ezafe.written`, marked sites | 51 | 96.1% | always ezāfe 96.1% |

The vowels are the Tajik edition's (یَکی، سَر، و read u), and Tajik doesn't
tell i from ī or u from ū. The prose profile fails on the verse: a silent-ه
head before a noun with no ezāfe is the usual case there (همه کوه، تیره خاک),
and it finds one at 252 of 508 such junctions. I tuned the verse profile on the
other four fifths of the poems (11,417 sites), where it gets 97.1% against 90.5%
for always none. In this metre an ezāfe after a silent ه is always long, so the
verse profile doesn't count lengthening it as a licence.

On modern verse it doesn't help. I labelled 252 unmarked junctions from Ganjoor,
the whole poem given as the text: 194 from Nima, Sepehri, Forough, Akhavan and
Shamlou, 58 from Hafez, Saadi and Rumi. I set the 40% from unlabelled poems,
before scoring these once.

| | sites | verse | always none |
|---|---:|---:|---:|
| Hafez, Saadi, Rumi | 58 | 98.3% | 91.4% |
| Nima, Sepehri, Forough, Akhavan, Shamlou | 194 | 88.1% | 91.2% |

Nimaic verse keeps a foot but not a line length, so a third of these junctions
are in poems that still agree on some classical metre. There it finds 5 ezāfes
that aren't there, and of the 17 that are it finds 1. The classical rows are the same with or without Ganjoor's
metre.

```sh
ghazal-nlp shahnameh-eval
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
| compounds (پایان‌نامه، بین‌المللی) | its printed lists | joined where careful writers mostly join them, in at least two kinds of source (`data/ghazal-compounds.tsv`); `house`, since the spaced form is in use too |
| pairs written both ways (آنها/آن‌ها) | its printed lists | what careful writers write more, pair by pair (`data/ghazal-house-forms.tsv`) |
| tanvin (واقعاً) | required | left as written |
| preferred spellings (اطاق/اتاق) | the Academy's pick | both fine |
| می before a verb | ZWNJ; spaced form only | ZWNJ; spaced or joined, checked by `verb` |
| digits, «», …, Latin , ; ? | not covered | Persian digits, « », …, ، ؛ ؟ |

The silent-ه list (`data/orthography-silent-he.tsv`, 1,253 words) comes from
how each word's own spellings treat the ه (خانه‌ای vs گرهی).

On [PerSpellData](https://github.com/rominaoji/PerSpellData)'s word-boundary
errors, the sentence pairs that differ only in where a space or ZWNJ goes. A
finding counts only if it changes nothing but separators. `exact` needs the
same separator as the correct side; `apart` only whether the letters are kept
apart, since PerSpellData often has a space where the house has ZWNJ (شده اند،
می روم).

| Virastman logs, 127,120 pairs, 138,503 edits | `ghazal` | `academy` |
|---|---:|---:|
| changes proposed | 24,513 | 16,246 |
| precision, exact / apart | 19.1% / 73.0% | 17.5% / 42.0% |
| recall, exact / apart | 3.4% / 4.3% | 2.1% / 2.2% |
| missing ZWNJ (میروم), recall | 30.9% | 16.6% |

Recall is low because 83% of the edits split two words written together
(تلاشکردند), which no rule looks for.

```sh
ghazal-nlp check "آنها کتاب ها را میخوانند"
ghazal-nlp check --style academy "آنها کتاب ها را میخوانند"
ghazal-nlp rules [--style academy]
ghazal-nlp verify
ghazal-nlp orthography-eval wrong_virastman.txt correct_virastman.txt [--style academy]
```

### `arabic`: Arabic inside Persian

Finds Arabic inside Persian text, conservatively, as two kinds of span. A
`passage` is Arabic syntax: a shared word, a name or a ة isn't enough, and
Persian words end a span. A `phrase` is a formula or set phrase (علیه السلام،
مع ذلک) or a preposition or vocative before an Arabic article (فی الحال، ایها
الامیر). Finding nothing means "don't know", not "no Arabic". A third, exact
matcher finds Quran wording, given a copy of the text (Tanzil format).

Tested by me on spans and lines from Wikipedia, Wikisource and Ganjoor
(`data/arabic-*-lab.jsonl`). Of 71 passages it found, 59 are Arabic with the
right boundaries and 67 are Arabic (83% and 94%). Of 55 lines I marked as
holding Arabic (two words or more), it finds something in 30, and in 1 of 27
lines I marked as holding none (ایها القاضی). I added phrases after looking
at those lines, so the line numbers are dev numbers.

```sh
ghazal-nlp arabic "چنانکه گفته‌اند الحمد لله رب العالمین و ..."
ghazal-nlp arabic-eval
```

## Data

The code is MIT. The data in `ghazal_nlp/data` is CC BY 4.0, except the files
that quote Wikipedia or Wikisource, which are CC BY-SA 4.0 (`data/LICENSE`
lists them); each TSV's header says where it comes from. Ganjoor,
the treebanks, SentenceBench, HomoRich, UniMorph, PerSpellData and
Ganjoor-Rhythm-Bench are only read for testing and aren't included.

The test data, one item per line (my labels, except the last):

- `ezafe-written-dev.jsonl`, `ezafe-junctions-dev.jsonl` and `ezafe-lab-1.jsonl`:
  silent-ه junctions. The text around the junction with its mark removed
  (`neutralized_text`, `neutral_head_span`, `neutral_next_span`), the mark
  that was there (`observed_mark_class`), `blind` (my label without seeing the
  mark) and, in the dev files, `revised` (my final label).
- `arabic-spans-lab.jsonl`: a span the detector found, and whether it's
  Arabic (`arabic`, `boundary` when the edges are off, `not_arabic`).
- `arabic-lines-lab.jsonl`: whether a line holds Arabic of two words or more.
- `shahnameh-tajik-test.jsonl`: Shāhnāme hemistichs with each word's vowels and
  ezāfe from the Tajik copy (ArabovMK/shahnameh-tajik-corpus, CC BY-SA 4.0).

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
- [Ganjoor-Rhythm-Bench](https://huggingface.co/datasets/farbodbij/Ganjoor-Rhythm-Bench), and Theodore S. Beers's [sangin](https://github.com/theodore-s-beers/sangin), the XLM-R metre classifier.
- Mahta Fetrat Qharabagh, Zahra Dehghanian and Hamid R. Rabiee, [Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models](https://arxiv.org/abs/2505.12973), 2025 (HomoRich), and [LLM-Powered Grapheme-to-Phoneme Conversion: Benchmark and Case Study](https://huggingface.co/datasets/MahtaFetrat/SentenceBench), ICASSP 2025 (SentenceBench).
- [UniMorph](https://unimorph.github.io/), Persian paradigms from Wiktionary.
- Romina Oji, Nasrin Taghizadeh and Heshaam Faili, [PerSpellData](https://aclanthology.org/2021.nsurl-1.2), NSURL 2021.

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
