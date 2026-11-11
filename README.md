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
fit; among equals, the one with the lowest licence cost and more poems behind
it, where each licence has its own cost, fitted on held-out poems that no number
below uses. Honorifics like (ع) are left out of the scan.

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
| gold metre among the answers | 99.1% | 95.4% |
| gold metre among the cheapest | 68.4% | 77.2% |
| metres per hemistich | 2.19 | 1.43 |
| poem gets exactly the gold metre | 93.4% | 96.4% |
| abstained | 0.07% | 0.07% |

The weight lexicon (`data/weights.tsv`) is learned from the labels of the
other four fifths, so that column is supervised. The rules-only column isn't,
apart from the 23 licence costs.
With the editors' marks removed (`--stripped`), a poem gets exactly the gold
metre 93.0% of the time with rules only and 96.5% with the lexicon.
Holding out whole poets instead of poems (`--by poet`, 2,000 poems by poets
the lexicon never saw), that's 92.5% and 96.0%.

A few lines are enough for most poems. On 2,000 other held-out poems (the next 2,000 in
the same shuffle), reading only their opening lines, the exact metre comes out:

| hemistichs given | 1 | 2 | 3 | 4 | 8 | whole poem |
|---|---:|---:|---:|---:|---:|---:|
| exact metre | 84.0% | 91.6% | 93.6% | 94.6% | 95.7% | 96.1% |
| wrong | 16.0% | 8.4% | 6.4% | 5.4% | 4.3% | 3.9% |

On single couplets with no poem around them it's much weaker. On the 12,713
couplets of [Ganjoor-Rhythm-Bench](https://huggingface.co/datasets/farbodbij/Ganjoor-Rhythm-Bench)
(40 metres), the rules alone give exactly the gold metre for 56.6%. With the
lexicon, on the 2,553 couplets from held-out poems, that's 72.8%.

On those couplets, next to [an XLM-R metre classifier](https://huggingface.co/katomyomachia/xlm-roberta-persian-meter)
(35 metres, trained on Ganjoor hemistichs; its card reports 98% on a tenth of
its own data):

| exact metre per couplet | couplets | XLM-R | `aruz` with the lexicon | always the commonest metre |
|---|---:|---:|---:|---:|
| all held-out | 2,553 | 61.9% | 72.8% | 4.0% |
| metre the model knows | 2,204 | 71.6% | 76.5% | 4.6% |
| … and no hemistich in its training data | 2,041 | 70.9% | 76.2% | 4.3% |

On lone couplets in its own metres `aruz` is about 5 points ahead, and across all
metres about 11, since 13.7% of the couplets are in metres the classifier can't
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
Shamisa, but he finds only a few in Hafez, so it over-marks. This paragraph and
the table above it were measured on 0.7.0.

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

### `vowels`: readings and vowel marks

Every reading of a word I could find, from editors' marks (Ganjoor, Wikipedia,
Wikisource, books), dictionaries (Dehkhoda, Steingass, Wiktionary) and
pronounced sentences (HomoRich's training part), with counts and sources for
each. `choose` picks the reading a sentence needs; `mark` writes the vowels
back, all it is sure of (`full`) or only those that tell a word from its
homographs (`minimal`), keeping the spelling and spacing. It writes short
vowels, distinguishing sukun, shadda and tanvin, without redundant marks for
long vowels. In verse it can use the metre. Ezāfe is optional (`ezafe=True`).

```python
from ghazal_nlp import vowels
vowels.lookup("گل")           # readings with counts and sources
vowels.choose("گل در باغ رویید.")
vowels.mark("ملک را به ملک بخشید.", mode="minimal")
```

Tested by me, with the vowels hidden first:

| | words | the commonest reading | `choose` |
|---|---:|---:|---:|
| HomoRich test, fixed homograph set | 66,108 | 57.0% | 95.7% |
| … with no training sentence above 0.5 similarity | 56,914 | 57.0% | 95.4% |
| … with no training sentence above 0.8 similarity | 65,656 | 57.0% | 95.7% |
| SentenceBench, homographs | 164 | 46.9% | 84.1% |
| 122 sentences I labelled, prose and verse | 143 | 49.6% | 76.9% |
| Wikipedia prose dev: every editor's mark recovered | 7,758 | 54.8% | 58.8% |
| Ganjoor, 1,891 dev poems: every editor's mark recovered | 21,885 | 73.9% | 72.6% |
| 150 common homographs in held-out sentences (Wikipedia, blogs, news, poetry) | 297 | 36.0% | 93.6% |

In the last row each sentence was read by two labellers who couldn't see each
other's answers, and the commonest reading leaves out HomoRich's counts, which
are balanced across readings by design.

`minimal` uses 106 reviewed homographs, a second reading with at least 5% of
the lexical or context evidence, and calibrated confidence of 0.8. Function
words ordinarily stay bare. On HomoRich, 96.3% of its marks on labelled
homographs are right, and it marks 22.0% of their occurrences (121,928 located
targets; 7,678 rows have no locatable target). On my labelled set these figures
are 91.3% and 65.0%. Marks elsewhere earn no credit. In prose and poetry I can
score only the slots the editors marked; unmarked slots are unknown. Modern
prose and unfamiliar contexts remain weak, and on Ganjoor the commonest reading
still wins.

### `ezafe`: the ezāfe after a silent ه

`junctions` finds every word from the silent-ه list that's followed by another word,
records how the junction is written (هٔ، ۀ، ه‌ی، هی، ه ی or nothing), and
removes just that mark. Two classifiers then look at the unmarked text:
`written` decides whether a written mark is an ezāfe or an old spelling of ‌ای,
and `junction` decides whether an unmarked junction has an ezāfe. A line break
counts as a boundary.

```python
from ghazal_nlp import ezafe
for j in ezafe.junctions("نامه را نوشتم و خانه زیبا بود"):
    print(j.head, j.mark, j.label, j.reason)
```

`junction` starts with rules for where there's no ezāfe: across punctuation,
before a verb, a copula, a connector or a preposition, before a plural ها or a
می written apart, and after heads that take none (همیشه، چگونه). The rest,
mostly a content word next, go to 150 small boosted trees over corpus counts:
how often the pair turns up, how often writers who mark the ezāfe mark it on
that pair, what usually follows the next word, and the words around. They
trained on 269,349 junctions from the corpus's Starter tables labelled by Hazm's
tagger and 1,257 labelled by hand, mine and those Claude Opus and GPT agreed on.

`junction` has two profiles. `prose`, the default, is the one above. `verse`
scans the line once with the junction read as an ezāfe and once without, and
takes whichever needs fewer licences. Where the metre says none, or can't
choose, it's none; where it says ezāfe, or there's no metre, the prose trees
decide. Give it the poem's metre when you know it; otherwise it takes the metre
most of the text's hemistichs scan cheapest in, if there are two or more and 40%
of them agree. I give it the whole poem.

```python
ezafe.junctions("فرستاده شاه را پیش خواند", profile="verse",
                metre="فعولن فعولن فعولن فعل (متقارب مثمن محذوف یا وزن شاهنامه)")
```

I check it on labelled sites and on SentenceBench and HomoRich pronunciations:

| | sites | agree | always the commoner label |
|---|---:|---:|---:|
| `written`, dev | 102 | 100 (98%) | 88.2% |
| `written`, first test | 78 | 73 (93.6%) | 87.2% |
| `written`, second test | 48 | 46 (95.8%) | 95.8% |
| `written`, SentenceBench | 14 | 13 (93%) | 100% |
| `written`, HomoRich | 2,724 | 98.5% | 98.9% |
| `junction`, prose, 5 folds | 524 | 480 (91.6%) | 66.8% |
| `junction`, prose, random sample, 5 folds | 149 | 137 (91.9%) | 73.2% |
| `junction`, verse | 252 | 238 (94.4%) | 91.3% |
| `junction`, SentenceBench | 132 | 116 (88%) | 76.5% |
| `junction`, HomoRich | 58,947 | 89.8% | 73.6% |

The `written` dev sites (`data/ezafe-written-dev.jsonl`) were labelled while I
wrote the rules, and I changed the rules after looking at the first test
(`data/ezafe-lab-1.jsonl`), so only the second is clean. `written` beats always
saying ezāfe on the first and ties on the second.

Every hand label I have for `junction` is in its training now, so its rows are
cross-validated: the trees are retrained five times, each time without a fifth of
the documents, and scored on that fifth. The 524 are my prose labels, many picked
where the old rules were unsure; 149 of them are a random sample of unmarked
junctions from the Starter tables. I set the verse rule on the 252 Ganjoor
junctions, so that row isn't a clean test. HomoRich's ezāfe marks were added
automatically (a POS tagger and a dictionary, per its paper), and about 40% of its
sentences GPT-4o wrote, so its rows measure agreement with that tagging, not with people.

`sites()` only sees words on its list, which on samples from seven tables is
54–67% of the ه-final words that aren't verbs.

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
follows each) and the words around, trained on kasras rewritten as ه. Another 300
trees, trained on 13,203 sites labelled by hand, correct them; they also see how
informal the 40 characters on either side read (`informality`). The labels are
mine and AI models': Claude's, and Claude Opus's and GPT's where both agreed.

It flags a site at probability 0.9; `threshold` changes that. On 726 labelled
sites from about 217,000 in spoken text, blogs, web, news, books and verse,
picked across the range of scores and weighted back, under five folds:

| | precision | lower 95% bound | share of the errors flagged |
|---|---:|---:|---:|
| threshold 0.9 | 1.00 | 0.88 | 0.15 |
| threshold 0.8 | 0.92 | 0.81 | 0.22 |
| the opt-in model below, 0.9 | 0.97 | 0.91 | 0.44 |
| the opt-in model below, 0.8 | 0.90 | 0.84 | 0.55 |
| 0.5.0 | 0.55 | | 0.18 |

The ezāfe written as ه is rare there, 0.24% of sites. The share is an upper
bound: among the lowest-scored sites I found none in 200, which doesn't rule out
a few. It still misses plenty, including the first example above.

```sh
ghazal-nlp ezafe-heh "اسمه من علیه"
```

### `neural`: the opt-in model

One TookaBERT fine-tuned for both decisions: first on 1,050,349 sites (kasras
rewritten as ه, and junctions labelled by Hazm's tagger), then on the hand
labels. For `heh`, trees over the same features correct its scores. It needs
torch and transformers (`pip install ghazal-nlp[neural]`) and the model's folder,
which I'll publish after a legal review.

```python
from ghazal_nlp.neural import Ezafe
model = Ezafe("path/to/model")
model.junctions("نامه را نوشتم و خانه زیبا بود")
model.heh("دیواره اتاقه من", threshold=0.9)
```

`junctions` keeps the rules and puts the model where the trees would decide. On
the unwritten ezāfe it's no better than the trees: 90.5% on the prose labels and
93.3% on the random sample under five folds, against 91.6% and 91.9%. On the
ezāfe written as ه it finds about three times as many (table above).

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
| `ezafe.junction`, prose | 561 | 71.8% | always none 90.6% |
| `ezafe.junction`, verse, the metre given | 561 | 96.4% | always none 90.6% |
| `ezafe.written`, marked sites | 51 | 96.1% | always ezāfe 96.1% |

The vowels are the Tajik edition's (یَکی، سَر، و read u), and Tajik doesn't
tell i from ī or u from ū. The prose profile fails on the verse: a silent-ه
head before a noun with no ezāfe is the usual case there (همه کوه، تیره خاک),
and it finds one at 152 of 508 such junctions. I tuned the verse profile on the
other four fifths of the poems (11,417 sites), where it gets 97.1% against 90.5%
for always none. In this metre an ezāfe after a silent ه is always long, so the
verse profile doesn't count lengthening it as a licence.

The 252 Ganjoor junctions above, the whole poem given as the text:

| | sites | agree | always none |
|---|---:|---:|---:|
| classical sources | 58 | 56 (96.6%) | 91.4% |
| modern verse | 194 | 182 (93.8%) | 91.2% |

On modern verse it finds 10 of the 17 real ezāfes and adds 5 that aren't there.
Nimaic verse keeps a foot but not a line length, so some poems agree on the wrong
classical metre, and there the trees decide.

```sh
ghazal-nlp shahnameh-eval
```

### `orthography`: spelling rules

Two styles: `ghazal`, my house spelling, and `academy`, the Academy's دستور خطّ
فارسی with a page for each rule. A finding is a `violation` (no careful writer
does it: Arabic ي, a space before ها, می written apart), `house` (both are
fine, here's mine) or `advisory` (a likely error I'm less sure of, shown only
with `min_confidence="low"`). Every rule is a TOML record in `rules/` with the
registers it applies to and examples it must and must not flag; a case it
can't decide is left alone.

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
| compounds (پایان‌نامه، بین‌المللی) | its printed lists | two-part joins where careful writers mostly join them (house); three parts stay apart |
| tanvin (واقعاً) | required | left as written |
| preferred spellings (اطاق/اتاق) | the Academy's pick | both fine |
| می before a verb | ZWNJ; spaced form only | ZWNJ; spaced or joined, checked by `verb` |
| two words run together (خواهندکرد) | not covered | `advisory` |
| digits, «», …, Latin , ; ? | not covered | Persian digits, « », …, ، ؛ ؟ |

Arabic and names keep their ة; ة written for the ezāfe becomes ه‌ی (توسعة
اقتصادی → توسعه‌ی اقتصادی).

Tested by me on 386 findings from fresh text (news, Wikipedia, social media,
books, blogs), judged blind before seeing the suggestion and weighted to how
often each finding occurs. A finding counts only if my correction is exactly
its suggestion: `violation` 97.6%, `house` 92.8%, `advisory` 56% (19 findings).

On [PerSpellData](https://github.com/rominaoji/PerSpellData)'s word-boundary
pairs, `exact` needs the same separator as the correct side, `apart` only that
the letters are kept apart (PerSpellData often has a space where I write ZWNJ):

| `ghazal` | precision, exact / apart | recall, exact / apart |
|---|---:|---:|
| Virastman, 95,353 pairs | 21.0% / 73.8% | 3.7% / 4.7% |
| … with advisories | 37.4% / 77.8% | 8.6% / 9.6% |
| the rest, 56,916 pairs | 47.0% / 77.4% | 9.4% / 9.5% |

```sh
ghazal-nlp check "آنها کتاب ها را میخوانند"
ghazal-nlp check --style academy --min-confidence low "..."
ghazal-nlp rules [--style academy]
ghazal-nlp verify
ghazal-nlp orthography-eval wrong_virastman.txt correct_virastman.txt
```

### `arabic`: Arabic inside Persian

Finds Arabic inside Persian text: a probability for every word, and spans of
Arabic syntax two words or longer (a Quran verse, a hadith, an Arabic
hemistich, a formula like علیه السلام, a set phrase like فی الحال, a title).
A loanword in Persian syntax is Persian however Arabic it looks (علم، ثلث،
مع‌الوصف). Standard library only.

```python
from ghazal_nlp import arabic

arabic.spans("او گفت: «هذا هو الذی لم یکن له شریک» و رفت.")
arabic.mask("علم و ثلث را به کار برد", strict=True)  # all False
```

`strict=True` is for a normalizer: a higher threshold, and the old rules have
to agree. `quran=QuranIndex(paths)` adds exact Quran wording from a local
Tanzil copy, which doesn't ship. `detector()` and the CLI are the old rules.

Tested by me on 275 lines from six kinds of text (classical prose, verse, wiki,
news, web, social), held out from everything the model was built on, and
weighted to how often each kind of line occurs:

| | word precision | word recall | lines with Arabic found | false alarms per 1,000 lines |
|---|---:|---:|---:|---:|
| rules (0.6.0) | 82% | 39% | 59% | 0.07 |
| model | 85% | 62% | 72% | 0.65 |
| strict, with the Quran index | 87% | 45% | 62% | 0.07 |

```sh
ghazal-nlp arabic "چنانکه گفته‌اند الحمد لله رب العالمین و ..."
```

### `normalize`: the Ghazal Corpus normalizer

The normalizer behind the `normalized` column of the Ghazal Corpus, version
1.0.0, so other text can be put in the same form. It repairs what isn't a
choice and leaves the rest as written: Arabic letters and presentation forms,
invisible characters and ZWNJs that change nothing, Arabic-Indic digits,
spacing and punctuation, ة in Persian words, and the junctions careful writers
write with a ZWNJ (می before a verb, ها, an ending after ه, compounds they
mostly join). Code in ``` blocks or `backticks` stays as written. A poem keeps
ای apart after a word, since it's mostly the vocative.

This is a copy of 1.0.0. The corpus normalizer keeps changing; this one only
changes with a release here. It gives the same bytes as the corpus normalizer
on 10,917,739 rows of the corpus bench, every source and profile.

```python
from ghazal_nlp import normalize
normalize.normalize("كتاب ها را ميخوانم.")           # کتاب‌ها را می‌خوانم.
normalize.normalize("پندم مده ای دوست", verse=True)  # پندم مده ای دوست
normalize.house("آنها 20% بیشتر")                    # آن‌ها ۲۰٪ بیشتر: my house forms where Persian has two right ones
normalize.key("خانهٔ پدری")                          # خانه پدری: every surface difference folded, for search and duplicates
normalize.equivalent("خانهٔ پدری", "خانه‌ی پدری")      # True: equal up to accepted variants
normalize.profile("ckb", "document")                 # "arab": the steps a corpus row in that language got (PROFILES)
```

## Data

The code is MIT. The data in `ghazal_nlp/data` is CC BY 4.0, except the files
that quote Wikipedia or Wikisource, which are CC BY-SA 4.0 (`data/LICENSE`
lists them); each TSV's header says where it comes from. `ezafe-junction-counts.tsv.gz`
and `ezafe-junction-trees.json` are `junction`'s counts and trees. `data/normalize` holds
the normalizer's tables as the corpus has them, counted on the corpus.
The Arabic model also derives statistics and parameters from Ganjoor and the
sources listed in its provenance; it is CC BY-SA 4.0.
The vowel tables and models are also CC BY-SA 4.0; HomoRich's recommended
training split supplies word pronunciations and context labels. The treebanks,
SentenceBench, UniMorph, PerSpellData and
Ganjoor-Rhythm-Bench are only read for testing and aren't included.

`data/corpus-sample.jsonl.gz` is the Ghazal Corpus 1.0.0 Sample, 500 documents
(50 from each Starter table) with `text` and `normalized`, to try the tools on;
each row carries its own licence (`data/LICENSE`):

```python
import gzip, json
from importlib.resources import files
rows = [json.loads(line) for line in gzip.open(files("ghazal_nlp") / "data/corpus-sample.jsonl.gz")]
```

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
