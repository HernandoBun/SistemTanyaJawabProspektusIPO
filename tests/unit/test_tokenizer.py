"""
tests/unit/test_tokenizer.py
Uji _normalize_numbers() dan _tokenize() dari src/retrieval/indexer.py
Jalankan dengan: python -m pytest tests/unit/test_tokenizer.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retrieval.tokenizer import _normalize_numbers, _tokenize

def test_thousand_separator():
    assert "1250000" in _tokenize("1.250.000")
    assert "1250000000" in _tokenize("1.250.000.000")

def test_rp_prefix():
    tokens = _tokenize("Rp 780.000 juta")
    assert "rp780000" in tokens or "780000" in tokens

def test_percentage():
    assert "18_7_persen" in _tokenize("meningkat 18,7%")

def test_integer_percentage():
    assert "25_persen" in _tokenize("sebesar 25%")

def test_plain_text_unchanged():
    tokens = _tokenize("pendapatan usaha perseroan")
    assert "pendapatan" in tokens and "usaha" in tokens

def test_combined():
    tokens = _tokenize("Total pendapatan Rp 1.250.000.000 naik 18,7%")
    assert "1250000000" in tokens
    assert "18_7_persen" in tokens

def test_unicode_letters_are_preserved():
    tokens = _tokenize("José García menjabat sebagai komisaris")
    assert "josé" in tokens
    assert "garcía" in tokens

if __name__ == "__main__":
    test_thousand_separator(); test_rp_prefix(); test_percentage()
    test_integer_percentage(); test_plain_text_unchanged(); test_combined()
    print("All tokenizer tests PASSED")
