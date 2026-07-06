#!/usr/bin/env python3
"""
BPE Tokenizer Training Pipeline
Trains a multilingual BPE tokenizer on English, Hindi, Telugu, and Tamil
Wikipedia "India" articles.
"""

import unicodedata
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFC

BASE = "/Users/umittal/Desktop/Tokenizer"

# ── Step 1: Load and NFC-normalize all texts ──────────────────────────────────
lang_files = {
    "English": f"{BASE}/india_en.md",
    "Hindi":   f"{BASE}/india_hi.md",
    "Telugu":  f"{BASE}/india_te.md",
    "Tamil":   f"{BASE}/india_ta.md",
}

texts = {}
for lang, path in lang_files.items():
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    texts[lang] = unicodedata.normalize("NFC", raw)
    print(f"Loaded {lang}: {len(texts[lang])} chars")

# ── Step 2: Build weighted training corpus ────────────────────────────────────
# Weights chosen to minimise fertility gap while keeping English fertility ≤ 1.6
WEIGHTS = {
    "English": 5,
    "Hindi":   4,
    "Telugu":  10,
    "Tamil":   5,
}

def build_corpus(weights):
    corpus = []
    for lang, w in weights.items():
        lines = texts[lang].splitlines()
        lines = [l for l in lines if l.strip()]
        corpus.extend(lines * w)
    return corpus

corpus = build_corpus(WEIGHTS)
print(f"\nTraining corpus: {len(corpus)} lines")

# Write corpus to a temp file for the trainer
corpus_path = f"{BASE}/train_corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    f.write("\n".join(corpus))
print(f"Corpus written to {corpus_path}")

# ── Step 3: Train BPE tokenizer ───────────────────────────────────────────────
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.normalizer = NFC()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
)

print("\nTraining tokenizer...")
tokenizer.train([corpus_path], trainer)

# Save tokenizer
save_dir = f"{BASE}/tokenizer"
os.makedirs(save_dir, exist_ok=True)
tokenizer.save(f"{save_dir}/tokenizer.json")
tokenizer.model.save(save_dir)
print(f"Tokenizer saved to {save_dir}/")

# Verify vocab size
vocab = tokenizer.get_vocab()
print(f"\nVocab size: {len(vocab)}")
assert len(vocab) == VOCAB_SIZE, f"Expected {VOCAB_SIZE}, got {len(vocab)}"

# ── Step 4: Compute fertility ─────────────────────────────────────────────────
def fertility(text, tok):
    words = text.split()
    if not words:
        return 0.0
    encoding = tok.encode(text)
    return len(encoding.tokens) / len(words)

print("\n" + "="*55)
print(f"{'Language':<10} {'Words':>8} {'Tokens':>8} {'Fertility':>10}")
print("-"*55)

results = {}
for lang, text in texts.items():
    words = text.split()
    enc = tokenizer.encode(text)
    fert = len(enc.tokens) / len(words)
    results[lang] = {"words": len(words), "tokens": len(enc.tokens), "fertility": fert}
    print(f"{lang:<10} {len(words):>8} {len(enc.tokens):>8} {fert:>10.4f}")

print("-"*55)
fertilities = [v["fertility"] for v in results.values()]
min_f = min(fertilities)
max_f = max(fertilities)
gap   = max_f - min_f
score = 1000 / gap if gap > 0 else float("inf")
print(f"\nMin fertility : {min_f:.4f}")
print(f"Max fertility : {max_f:.4f}")
print(f"Gap           : {gap:.4f}")
print(f"Score (1000/gap): {score:.2f}")

en_fert = results["English"]["fertility"]
constraint = "PASS" if en_fert <= 1.6 else "FAIL"
print(f"\nEnglish fertility ≤ 1.6: {constraint}  (actual: {en_fert:.4f})")
print("="*55)

# Save weights used
weights_path = f"{BASE}/weights_used.txt"
with open(weights_path, "w") as f:
    f.write("Final weights:\n")
    for k, v in WEIGHTS.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nFertility results:\n")
    for lang, r in results.items():
        f.write(f"  {lang}: words={r['words']}, tokens={r['tokens']}, fertility={r['fertility']:.4f}\n")
    f.write(f"\nGap: {gap:.4f}  Score: {score:.2f}\n")
    f.write(f"English constraint (≤1.6): {constraint}\n")
print(f"\nWeights/results saved to {weights_path}")
