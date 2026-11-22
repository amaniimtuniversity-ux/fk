"""MMS Hindi pronunciation normaliser: nothing speakable may be dropped, no empty tensors."""
from __future__ import annotations

import re

from app.hindi_text import has_speakable_devanagari, normalize_for_mms, number_to_hindi


def test_numbers():
    assert number_to_hindi(12) == "बारह"
    assert number_to_hindi(2024) == "दो हज़ार चौबीस"
    assert number_to_hindi(250000) == "दो लाख पचास हज़ार"
    assert normalize_for_mms("रात के 12:30 बजे") == "रात के बारह तीस बजे"
    assert normalize_for_mms("₹500 aur 75%") == "पाँच सौ रुपये और पचहत्तर प्रतिशत"
    assert "९" not in normalize_for_mms("सन् २०२४")


def test_latin_sentence_becomes_devanagari():
    out = normalize_for_mms("Ramesh ne darwaza khola.")
    assert not re.search(r"[A-Za-z]", out)
    assert has_speakable_devanagari(out)
    assert "दरवाज़ा" in out and "खोला" in out


def test_loanwords_and_acronyms():
    out = normalize_for_mms("Police station par CID aayi, video viral hua.")
    assert "पुलिस स्टेशन" in out and "सी आई डी" in out and "वीडियो" in out and out.endswith("हुआ.")


def test_only_punctuation_is_detected_not_crashed():
    out = normalize_for_mms("...")
    assert not has_speakable_devanagari(out)


def test_pure_devanagari_untouched():
    s = "अचानक पीछे से किसी ने उसका नाम पुकारा।"
    assert normalize_for_mms(s) == s
