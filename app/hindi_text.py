"""Pronunciation normalisation for the MMS Hindi TTS (facebook/mms-tts-hin).

Why this exists (seen on Kaggle T4):
  The MMS Hindi tokenizer knows ONLY Devanagari letters (and a few digits). Latin text, "...",
  symbols and most digits are silently DROPPED. A sentence that is entirely Latin/Hinglish
  tokenises to an *empty float tensor* -> "Expected tensor for argument #1 'indices' ... got
  torch.cuda.FloatTensor" and the whole job dies. Mixed sentences lose words silently (script
  no longer spoken verbatim).

This module converts ONLY the copy that is sent to the synthesiser (subtitles/manifest keep the
original text):
  * ASCII/Devanagari digits -> Hindi number words (vocab lacks 5,6,7,9 anyway)
  * common English loanwords -> standard Devanagari spelling (small dictionary)
  * ALL-CAPS acronyms (OK, TV, CID) -> letter names (ओ के, टी वी, सी आई डी)
  * other Latin words -> treated as romanised Hindi (Hinglish) and transliterated
  * symbols (%, ₹, &, +, @, ...) -> words / pauses
Rule-based: English words outside the dictionary are approximated. For perfect English
pronunciation use the Indic Parler-TTS backend (handles Latin natively).
"""
from __future__ import annotations

import re

_ONES = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
         "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
         "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
         "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
         "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
         "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
         "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
         "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
         "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे",
         "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"]


def number_to_hindi(n: int) -> str:
    """Integer -> Hindi words (Indian system: हज़ार, लाख, करोड़)."""
    if n < 0:
        return "माइनस " + number_to_hindi(-n)
    if n < 100:
        return _ONES[n]
    parts: list[str] = []
    for value, name in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            q, n = divmod(n, value)
            parts.append((number_to_hindi(q) if q != 1 or value == 100 else "एक") + " " + name)
    if n:
        parts.append(_ONES[n])
    return " ".join(parts)


_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Common English loanwords in Hindi scripts -> conventional Devanagari spelling.
_LOANWORDS = {
    "ok": "ओके", "okay": "ओके", "yes": "यस", "no": "नो", "hello": "हैलो", "hi": "हाय", "bye": "बाय",
    "video": "वीडियो", "audio": "ऑडियो", "phone": "फ़ोन", "mobile": "मोबाइल", "camera": "कैमरा",
    "photo": "फ़ोटो", "computer": "कंप्यूटर", "laptop": "लैपटॉप", "internet": "इंटरनेट", "online": "ऑनलाइन",
    "police": "पुलिस", "station": "स्टेशन", "hospital": "हॉस्पिटल", "doctor": "डॉक्टर", "school": "स्कूल",
    "college": "कॉलेज", "office": "ऑफ़िस", "bank": "बैंक", "market": "मार्केट", "hotel": "होटल",
    "room": "रूम", "house": "हाउस", "home": "होम", "door": "डोर", "window": "विंडो", "road": "रोड",
    "car": "कार", "bus": "बस", "train": "ट्रेन", "bike": "बाइक", "taxi": "टैक्सी", "auto": "ऑटो",
    "time": "टाइम", "night": "नाइट", "day": "डे", "morning": "मॉर्निंग", "evening": "ईवनिंग",
    "friend": "फ़्रेंड", "friends": "फ़्रेंड्स", "family": "फ़ैमिली", "boss": "बॉस", "sir": "सर", "madam": "मैडम",
    "boy": "बॉय", "girl": "गर्ल", "baby": "बेबी", "uncle": "अंकल", "aunty": "आंटी", "papa": "पापा", "mummy": "मम्मी",
    "story": "स्टोरी", "horror": "हॉरर", "ghost": "घोस्ट", "zombie": "ज़ॉम्बी", "monster": "मॉन्स्टर",
    "game": "गेम", "games": "गेम्स", "movie": "मूवी", "film": "फ़िल्म", "song": "सॉन्ग", "music": "म्यूज़िक",
    "news": "न्यूज़", "channel": "चैनल", "youtube": "यूट्यूब", "subscribe": "सब्सक्राइब", "like": "लाइक",
    "share": "शेयर", "comment": "कमेंट", "follow": "फ़ॉलो", "live": "लाइव", "viral": "वायरल",
    "please": "प्लीज़", "sorry": "सॉरी", "thanks": "थैंक्स", "thank": "थैंक", "you": "यू", "welcome": "वेलकम",
    "start": "स्टार्ट", "stop": "स्टॉप", "end": "एंड", "final": "फ़ाइनल", "last": "लास्ट", "first": "फ़र्स्ट",
    "new": "न्यू", "old": "ओल्ड", "big": "बिग", "small": "स्मॉल", "best": "बेस्ट", "super": "सुपर",
    "power": "पावर", "light": "लाइट", "dark": "डार्क", "black": "ब्लैक", "white": "व्हाइट", "red": "रेड",
    "blue": "ब्लू", "green": "ग्रीन", "water": "वॉटर", "fire": "फ़ायर", "air": "एयर", "earth": "अर्थ",
    "money": "मनी", "cash": "कैश", "rupees": "रुपये", "rupee": "रुपया", "dollar": "डॉलर", "price": "प्राइस",
    "free": "फ़्री", "sale": "सेल", "offer": "ऑफ़र", "business": "बिज़नेस", "job": "जॉब", "work": "वर्क",
    "love": "लव", "life": "लाइफ़", "death": "डेथ", "god": "गॉड", "king": "किंग", "queen": "क्वीन",
    "india": "इंडिया", "indian": "इंडियन", "delhi": "दिल्ली", "mumbai": "मुंबई", "america": "अमेरिका",
    "english": "इंग्लिश", "hindi": "हिंदी", "science": "साइंस", "history": "हिस्ट्री", "mystery": "मिस्ट्री",
    "secret": "सीक्रेट", "fact": "फ़ैक्ट", "facts": "फ़ैक्ट्स", "truth": "ट्रुथ", "real": "रियल", "fake": "फ़ेक",
    "test": "टेस्ट", "check": "चेक", "problem": "प्रॉब्लम", "solution": "सॉल्यूशन", "plan": "प्लान",
    "system": "सिस्टम", "machine": "मशीन", "robot": "रोबोट", "space": "स्पेस", "planet": "प्लैनेट",
    "and": "ऐंड", "the": "द", "of": "ऑफ़", "in": "इन", "on": "ऑन", "to": "टू", "is": "इज़", "it": "इट",
    "one": "वन", "two": "टू", "three": "थ्री", "four": "फ़ोर", "five": "फ़ाइव", "ten": "टेन",
}

# Frequent romanised-Hindi (Hinglish) words whose rule-based transliteration would be off.
_HINGLISH_WORDS = {
    "mein": "में", "me": "में", "hai": "है", "hain": "हैं", "nahi": "नहीं", "nahin": "नहीं", "nhi": "नहीं",
    "kya": "क्या", "aur": "और", "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से", "ne": "ने",
    "par": "पर", "pe": "पे", "tha": "था", "thi": "थी", "the": "थे", "hua": "हुआ", "hui": "हुई", "hue": "हुए",
    "main": "मैं", "hum": "हम", "tum": "तुम", "aap": "आप", "woh": "वो", "wo": "वो", "yeh": "ये", "ye": "ये",
    "uska": "उसका", "uski": "उसकी", "uske": "उसके", "unka": "उनका", "mera": "मेरा", "meri": "मेरी",
    "bahut": "बहुत", "bohot": "बहुत", "kuch": "कुछ", "phir": "फिर", "fir": "फिर", "ab": "अब", "tab": "तब",
    "jab": "जब", "kyun": "क्यों", "kyon": "क्यों", "kaise": "कैसे", "kaun": "कौन", "kahan": "कहाँ",
    "yahan": "यहाँ", "wahan": "वहाँ", "sab": "सब", "bhi": "भी", "hi": "ही", "to": "तो", "toh": "तो",
    "lekin": "लेकिन", "magar": "मगर", "ek": "एक", "din": "दिन", "raat": "रात", "ghar": "घर", "log": "लोग",
    "baat": "बात", "dost": "दोस्त", "dar": "डर", "bhoot": "भूत", "andhera": "अंधेरा", "aawaz": "आवाज़",
    "awaaz": "आवाज़", "darwaza": "दरवाज़ा", "haveli": "हवेली", "achanak": "अचानक", "dheere": "धीरे",
    "andar": "अंदर", "bahar": "बाहर", "peeche": "पीछे", "aage": "आगे", "upar": "ऊपर", "neeche": "नीचे",
    "accha": "अच्छा", "acha": "अच्छा", "theek": "ठीक", "thik": "ठीक", "zaroor": "ज़रूर", "shayad": "शायद",
    "kahani": "कहानी", "kahaani": "कहानी", "zindagi": "ज़िंदगी", "duniya": "दुनिया", "pyaar": "प्यार",
    "dil": "दिल", "sapna": "सपना", "sach": "सच", "jhooth": "झूठ", "waqt": "वक़्त", "samay": "समय",
    "aaj": "आज", "kal": "कल", "subah": "सुबह", "shaam": "शाम", "dopahar": "दोपहर", "aadmi": "आदमी",
    "aurat": "औरत", "bachcha": "बच्चा", "bacche": "बच्चे", "ladka": "लड़का", "ladki": "लड़की",
    "gaon": "गाँव", "shehar": "शहर", "sheher": "शहर", "jungle": "जंगल", "pahad": "पहाड़", "nadi": "नदी",
    "chalo": "चलो", "dekho": "देखो", "suno": "सुनो", "ruko": "रुको", "bhago": "भागो", "aao": "आओ", "jao": "जाओ",
    "mat": "मत", "kabhi": "कभी", "hamesha": "हमेशा", "koi": "कोई", "kisi": "किसी", "jisne": "जिसने",
    "liye": "लिए", "saath": "साथ", "bina": "बिना", "sirf": "सिर्फ़", "bas": "बस", "abhi": "अभी",
}

_LETTER_NAMES = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ़", "g": "जी", "h": "एच", "i": "आई",
    "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ", "p": "पी", "q": "क्यू", "r": "आर",
    "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू", "x": "एक्स", "y": "वाई", "z": "ज़ेड",
}

# Hinglish romanisation -> ITRANS-ish tweaks before transliteration.
_HINGLISH_FIXES = [
    (r"aa", "A"), (r"ee", "I"), (r"oo", "U"), (r"ei", "e"), (r"ai", "ai"), (r"au", "au"),
    (r"w", "v"), (r"ph", "f"), (r"ck", "k"), (r"q", "k"), (r"x", "ks"),
    (r"c(?=[eiy])", "s"), (r"c(?!h)", "k"),
]


def _romanised_to_devanagari(word: str) -> str:
    try:
        from indic_transliteration import sanscript
    except Exception:  # library missing -> spell out letters so nothing is dropped silently
        return " ".join(_LETTER_NAMES.get(ch, "") for ch in word.lower()).strip()
    w = word.lower()
    for pat, rep in _HINGLISH_FIXES:
        w = re.sub(pat, rep, w)
    w = re.sub(r"a$", "A", w)
    w = re.sub(r"i$", "I", w)
    w = re.sub(r"u$", "U", w)
    out = sanscript.transliterate(w, sanscript.ITRANS, sanscript.DEVANAGARI)
    # keep only characters the MMS vocab can speak (drop stray Latin leftovers)
    out = re.sub(r"[A-Za-z]", "", out)
    return out


def _latin_word(word: str) -> str:
    low = word.lower()
    if low in _HINGLISH_WORDS:
        return _HINGLISH_WORDS[low]
    if low in _LOANWORDS:
        return _LOANWORDS[low]
    if word.isupper() and 1 <= len(word) <= 5:        # acronym: OK, TV, CID, NASA -> letter names
        return " ".join(_LETTER_NAMES[c] for c in low if c in _LETTER_NAMES)
    if len(word) == 1:
        return _LETTER_NAMES.get(low, "")
    return _romanised_to_devanagari(word)


def _number_token(tok: str) -> str:
    tok = tok.translate(_DEV_DIGITS)
    if ":" in tok:                                      # 12:30 -> बारह तीस
        return " ".join(_number_token(p) for p in tok.split(":") if p)
    if "." in tok:                                      # 3.5 -> तीन दशमलव पाँच
        a, _, b = tok.partition(".")
        return (_number_token(a) if a else "शून्य") + " दशमलव " + " ".join(_ONES[int(d)] for d in b if d.isdigit())
    digits = re.sub(r"\D", "", tok)
    if not digits:
        return ""
    if len(digits) > 9 or (digits.startswith("0") and len(digits) > 1):   # phone numbers etc.: digit by digit
        return " ".join(_ONES[int(d)] for d in digits)
    return number_to_hindi(int(digits))


_TOKEN_RE = re.compile(r"[0-9०-९][0-9०-९,.:]*|[A-Za-z]+(?:'[A-Za-z]+)?|\u2026|\.{2,}|[%₹&+@#/]|Rs\.?")


def normalize_for_mms(text: str) -> str:
    """Return the pronunciation copy of `text` for MMS Hindi. Original text is never modified."""
    t = text.translate(_DEV_DIGITS)
    t = re.sub(r"(₹|Rs\.?)\s*([0-9][0-9,.]*)", lambda m: m.group(2) + " रुपये", t)      # ₹500 -> 500 रुपये

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0].isdigit():
            return " " + _number_token(tok.rstrip(",.")) + " "
        if tok in ("…",) or tok.startswith(".."):
            return ", "
        if tok == "%":
            return " प्रतिशत "
        if tok == "&":
            return " और "
        if tok == "+":
            return " प्लस "
        if tok == "@":
            return " ऐट "
        if tok in ("#", "/"):
            return " "
        if tok.lower().startswith("rs"):
            return " रुपये "
        return " " + _latin_word(tok) + " "

    t = _TOKEN_RE.sub(repl, t)
    t = re.sub(r"[\"“”‘’()\[\]{}<>*_~|\\^=]", " ", t)
    t = re.sub(r"\s+([,।.?!])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_speakable_devanagari(text: str) -> bool:
    return re.search(r"[\u0900-\u097F]", text) is not None
