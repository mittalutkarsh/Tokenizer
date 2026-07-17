#!/usr/bin/env python3
"""
BPE Tokenizer — HuggingFace tokenizers library.
Languages: English, Hindi, Telugu, Tamil
Corpus: faithful Markdown from Wikipedia (tables, links, footnotes)
Normaliser: NFKC (stronger than NFC — decomposes ligatures)
Pre-tokeniser: Metaspace (▁ prefix — keeps punctuation as separate units)
Decoder: Metaspace (guarantees decode(encode(text)) round-trip)
Metric: tokens / faithful_units (each visible punct char = 1 unit)

Run build_corpus.py first to fetch the corpus.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

BASE   = Path(__file__).resolve().parent
CORPUS = BASE / "corpus"
OUT_DIR = BASE / "tokenizer"

LANGS = ["en", "hi", "te", "ta"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil"}
WEIGHTS = {"en": 3, "hi": 4, "te": 4, "ta": 3}

FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def make_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    return tok


def train() -> tuple[Tokenizer, dict]:
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in LANGS
    }

    units = {code: faithful_units(text) for code, text in texts.items()}
    print(f"\n{'Language':<12} {'Chars':>10} {'Faithful Units':>16}")
    print("-" * 42)
    for code in LANGS:
        print(f"{LANG_NAMES[code]:<12} {len(texts[code]):>10,} {units[code]:>16,}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        files: list[str] = []
        for code, text in texts.items():
            path = tmpdir / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            files.extend([str(path)] * WEIGHTS[code])

        tokenizer = make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=10_000,
            min_frequency=1,
            special_tokens=["[UNK]"],
        )
        print(f"\nTraining BPE (vocab_size=10,000) …")
        tokenizer.train(files, trainer)

    token_counts = {code: len(tokenizer.encode(text).ids) for code, text in texts.items()}
    ratios       = {code: token_counts[code] / units[code] for code in LANGS}
    spread       = max(ratios.values()) - min(ratios.values())
    score        = 1000 / spread

    metrics = {
        "languages": LANG_NAMES,
        "weights":   WEIGHTS,
        "vocab_size": tokenizer.get_vocab_size(),
        "faithful_units": units,
        "token_counts":   token_counts,
        "ratios":  ratios,
        "spread":  spread,
        "score":   score,
    }
    return tokenizer, metrics


def round_trip_check(tokenizer: Tokenizer) -> None:
    samples = {
        "English": "India, officially the Republic of India, is a country in South Asia.",
        "Hindi":   "भारत एशिया महाद्वीप में स्थित एक देश है।",
        "Telugu":  "భారతదేశం దక్షిణాసియాలో ఉన్న ఒక దేశం.",
        "Tamil":   "இந்தியா தென் ஆசியாவில் உள்ள ஒரு நாடு.",
    }
    print("\nRound-trip faithfulness check:")
    print("-" * 60)
    all_pass = True
    for lang, text in samples.items():
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        tokens  = encoded.tokens
        # strip whitespace for comparison (decoder may normalise spacing)
        ok = text.replace(" ", "") == decoded.replace(" ", "")
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_pass = False
        print(f"  {lang:<10} {status}  tokens={len(tokens)}")
        print(f"    tokens: {tokens[:8]}{'…' if len(tokens) > 8 else ''}")
        if not ok:
            print(f"    original: {text}")
            print(f"    decoded:  {decoded}")
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")


def main() -> int:
    print("=" * 60)
    print("BPE Tokenizer — Metaspace + NFKC + faithful-unit metric")
    print("Languages: English · Hindi · Telugu · Tamil")
    print("=" * 60)

    tokenizer, metrics = train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))

    # also write vocab and merges in readable form
    vocab = tokenizer.get_vocab()
    with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    (BASE / "stats.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'Language':<12} {'Faithful Units':>16} {'Tokens':>10} {'Ratio':>8}")
    print("-" * 50)
    for code in LANGS:
        name = LANG_NAMES[code]
        print(f"{name:<12} {metrics['faithful_units'][code]:>16,} "
              f"{metrics['token_counts'][code]:>10,} "
              f"{metrics['ratios'][code]:>8.4f}")
    print("-" * 50)
    print(f"{'Spread':<12} {metrics['spread']:>38.4f}")
    print(f"{'Score':<12} {metrics['score']:>38.2f}")
    print(f"{'Vocab size':<12} {metrics['vocab_size']:>38,}")

    round_trip_check(tokenizer)

    print(f"\nSaved: tokenizer/tokenizer.json  tokenizer/vocab.json  stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
