"""Conservative word aligners. No match or multiple placements return no reading."""

import re
import unicodedata
from functools import lru_cache

from . import vowels as v

CONSONANTS = dict(zip("بپتطثسصجچحهخدذزضظرژشغقفکگلمن", "bpttsssjChhxdzzzzrZSqqfkglmn"))
SHORT = dict(zip("aeoiuA", [v.FATHA, v.KASRA, v.DAMMA, v.KASRA, v.DAMMA, v.FATHA]))


def notation(sound: str, scheme: str) -> str | None:
    if scheme == "homorich":
        sound = re.sub(r"(?:y?e)1$", "", sound).translate(str.maketrans({"/": "a", "a": "A", "$": "S", "c": "C", "@": "?", ";": "Z"}))
    elif scheme == "ipa":
        for a, b in [("d͡ʒ", "J"), ("t͡ʃ", "C"), ("dʒ", "J"), ("tʃ", "C"), ("aː", "A"), ("ɑː", "A"), ("ɒː", "A"), ("iː", "i"), ("uː", "u"), ("eː", "i"), ("oː", "u")]:
            sound = sound.replace(a, b)
        sound = sound.translate(str.maketrans({"æ":"a", "ä":"a", "ɑ":"A", "ɒ":"A", "ɐ":"a", "ɛ":"e", "ɪ":"i", "ʊ":"u", "ɔ":"o", "ʃ":"S", "ʒ":"Z", "ɣ":"q", "ɢ":"q", "ɡ":"g", "χ":"x", "ʔ":"?", "ɾ":"r", "ɹ":"r", "ɦ":"h", "j":"y"}))
        # IPA affricates were protected above; Persian ج is not IPA /j/.
        sound = sound.replace("J", "j")
        sound = re.sub(r"[\[\]/.ˈˌː]", "", sound)
        sound = "".join(c for c in unicodedata.normalize("NFD", sound) if not unicodedata.combining(c))
    elif scheme == "roman":
        for a,b in [("kh","x"),("gh","q"),("sh","S"),("zh","Z"),("ch","C"),("th","s"),("dh","z")]:
            sound=sound.replace(a,b)
        sound=sound.translate(str.maketrans({"ā":"A","â":"A","ī":"i","ū":"u","ḥ":"h","ṣ":"s","ẓ":"z","ṭ":"t","ḍ":"z","ẕ":"z","ḳ":"q","ġ":"q","š":"S","ž":"Z","č":"C","ʼ":"?","ʿ":"?","‘":"?","’":"?","'":"?","w":"v"}))
    elif scheme == "tajik":
        # ӯ merges historical vowels and cannot safely be mapped to Iranian Persian.
        if any(c in sound.lower() for c in "ӯыьцщэёюя"): return None
        sound=sound.lower().translate(str.maketrans(dict(zip("абвгғдеёзийкқлмнопрстуфхҳчҷшъӣ", "abvgqdeZziykqlmnAprstufxhCjS?i"))))
    elif scheme != "sound":
        raise ValueError(scheme)
    sound = sound.replace("-", "")
    return sound if sound and re.fullmatch(r"[abptdkgqfvszSZjCxhmnlry?eoiuA]+", sound) else None


@lru_cache(maxsize=65536)
def align_sound(word: str, sound: str, scheme: str = "sound", ow: bool = False) -> str | None:
    """``ow`` reads o before a medial و as the diphthong (fatḥa and sukun), not as long o."""
    word = v.skeleton(v.canonicalize(word))
    sound = notation(sound, scheme)
    if not sound or not word or len(word) > 40 or not all(c in CONSONANTS or c in "اآویعءئأإؤة‌" for c in word):
        return None

    @lru_cache(None)
    def walk(i, j, carrier=0):
        if i == len(word): return ("",) if j == len(sound) and not carrier else ()
        ch = word[i]
        if carrier:
            bases = [("", "")]
        elif ch in CONSONANTS:
            c = CONSONANTS[ch]
            bases = [(c, ""), (c+c, v.SHADDA)]
        elif ch == "‌": bases = [("", "")]
        elif ch == "آ": bases = [("A", ""), ("?A", "")]
        elif ch == "ا": bases = [("", ""), ("?", "")] if i == 0 else [("A", "")]
        elif ch == "و": bases = [(s, "") for s in ("v", "u", "o")] + [("vv", v.SHADDA)]
        elif ch == "ی": bases = [(s, "") for s in ("y", "i")] + [("yy", v.SHADDA)]
        elif ch == "ة": bases = [("t", ""), ("h", "")]
        else: bases = [("?", ""), ("", "")]
        out = set()
        for base, gem in bases:
            if not sound.startswith(base, j): continue
            k=j+len(base)
            vowel_options=[("", v.SUKUN, 0)] if carrier == 3 else [("", "", max(0, carrier - 1))]
            consonant = ch not in "‌آ" and not carrier and (not base or base[-1] not in "aeoiuA")
            if consonant:
                vowel_options=[("", v.SUKUN, False)]
                if k < len(sound) and sound[k] in SHORT:
                    c=sound[k]
                    long = i+1<len(word) and word[i+1] in {"A":"ا", "i":"ی", "u":"و", "o":"و"}.get(c, "")
                    long = long or (c in "ea" and i+2 == len(word) and word[i+1] == "ه")
                    long = 2 if c == "A" and ch == "خ" and word[i+1:i+3] == "وا" else int(long)
                    if long:
                        vowel_options = []
                    if c != "A":
                        vowel_options.append((c, SHORT[c], False))
                    if long:
                        diphthong = ow and c == "o" and ch != "خ" and i + 2 < len(word)
                        vowel_options.append((c, v.FATHA, 3) if diphthong else (c, SHORT[c], long))
            for consumed, mark, next_carrier in vowel_options:
                for tail in walk(i+1, k+len(consumed), next_carrier):
                    out.add(ch+mark+gem+tail)
                    if len(out)>2: return tuple(sorted(out))
        return tuple(sorted(out))
    ey = v.KASRA + "ی" + v.SUKUN
    forms={v.split_terminal(v.canonicalize(f.replace(ey, v.FATHA + "ی" + v.SUKUN)))[0] for f in walk(0,0)}
    return forms.pop() if len(forms)==1 else None


def align_letters(word: str, cue: str) -> tuple[str, ...]:
    """Dehkhoda's omitted letters stay unknown; accept only unique subsequences.

    A slash without a complete repeated skeleton has uncertain local scope.
    Tatweel and a free hamza have no unique carrier and are rejected.
    """
    word=v.skeleton(v.canonicalize(word))
    cue=cue.strip().strip("[]").strip()
    if not word or any(c.isspace() for c in word) or len(word)>40 or "ـ" in cue: return ()
    alternatives=cue.split("/")
    if len(alternatives)>1 and len({v.skeleton(v.canonicalize(re.sub(r"\s+", "", a))) for a in alternatives})!=1:
        return ()
    out=[]
    for alt in alternatives:
        parsed=v.slots(v.canonicalize(re.sub(r"\s+", "", alt)))
        if not parsed or any(c not in word or c=="ء" for c,m in parsed): return ()
        placements=[([],0)]
        for c,marks in parsed:
            placements=[(p+[(i,marks)],i+1) for p,start in placements for i in range(start,len(word)) if word[i]==c]
            if len(placements)>64: return ()
        forms=set()
        for placement,_ in placements:
            slots=[[c,""] for c in word]
            for i,m in placement: slots[i][1]=m
            form=v.split_terminal(v.canonicalize(v._render(slots)))[0]
            forms.add(form)
        if len(forms)!=1 or not v.is_marked(next(iter(forms))):return ()
        out.extend(forms)
    return tuple(sorted(set(out)))
