#!/usr/bin/env python3
"""
BPE Tokenizer — from scratch.
No HuggingFace tokenizers / sentencepiece / external ML libs.
Allowed: stdlib, unicodedata, re, collections, json.

Design: integer-ID word-frequency dict (best of BPETokenizerSimple +
word-freq deduplication).  Pair counting and merging run on tuples of
ints, which is ~3-5x faster than space-joined strings + regex.
"""

import unicodedata
import re
import json
import os
from collections import Counter, deque
from functools import lru_cache

BASE = "/Users/umittal/Desktop/Tokenizer"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# Corpus weights — tuned to minimise fertility gap while keeping English ≤ 1.6
WEIGHTS = {
    "English": 115,
    "Hindi":    80,
    "Telugu":  230,
    "Tamil":   130,
}

# Common English subword prefixes pre-seeded into vocab — free merges for English
# without competing in the BPE training loop.
ENGLISH_PRESEED = ["th", "he", "in", "er", "re", "on", "an", "en", "at", "or"]

# Number of top-frequency Telugu character bigrams to pre-seed (computed at runtime)
TELUGU_PRESEED_N = 10

LANG_FILES = {
    "English": f"{BASE}/india_en.md",
    "Hindi":   f"{BASE}/india_hi.md",
    "Telugu":  f"{BASE}/india_te.md",
    "Tamil":   f"{BASE}/india_ta.md",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def pretokenize_text(text):
    """
    Plain whitespace pre-tokeniser.
    Splits on any whitespace; each resulting chunk is one word token.
    Consistent with the fertility definition (words = whitespace-split).
    """
    return text.split()


def build_char_vocab(texts):
    """Collect every unique character across all texts; return sorted list."""
    chars = set()
    for text in texts.values():
        for tok in pretokenize_text(text):
            chars.update(tok)
    return sorted(chars)


# ── core BPE ──────────────────────────────────────────────────────────────────

def build_word_freq_dict(texts, weights, vocab_str2id):
    """
    Build {tuple_of_ids: frequency} from all languages, weighted.
    Each word is represented as a tuple of character-level token IDs.
    """
    word_freqs = Counter()
    for lang, text in texts.items():
        w = weights[lang]
        for tok in pretokenize_text(text):
            ids = tuple(vocab_str2id.get(ch, vocab_str2id["[UNK]"]) for ch in tok)
            word_freqs[ids] += w
    print(f"Unique word types in weighted corpus: {len(word_freqs):,}")
    return word_freqs


def get_pairs(word_freqs):
    """Count adjacent (id_a, id_b) pairs weighted by word frequency."""
    pairs = Counter()
    for ids, freq in word_freqs.items():
        for pair in zip(ids, ids[1:]):
            pairs[pair] += freq
    return pairs


def merge_word_freqs(word_freqs, pair, new_id):
    """
    Replace every occurrence of adjacent `pair` with `new_id` in every word.
    Uses deque-style linear scan (no regex) — O(n) per word.
    """
    a, b = pair
    new_freqs = {}
    for ids, freq in word_freqs.items():
        if a not in ids:          # fast skip — avoids full scan
            new_freqs[ids] = freq
            continue
        dq = deque(ids)
        result = []
        while dq:
            cur = dq.popleft()
            if cur == a and dq and dq[0] == b:
                result.append(new_id)
                dq.popleft()
            else:
                result.append(cur)
        new_freqs[tuple(result)] = freq
    return new_freqs


def train_bpe(texts, weights, vocab_size):
    print("Building character vocabulary …")
    chars = build_char_vocab(texts)

    # Assign IDs: special tokens first, then sorted characters
    vocab_id2str = {}
    vocab_str2id = {}
    for tok in SPECIAL_TOKENS:
        i = len(vocab_id2str)
        vocab_id2str[i] = tok
        vocab_str2id[tok] = i
    if "Ġ" not in vocab_str2id:
        i = len(vocab_id2str)
        vocab_id2str[i] = "Ġ"
        vocab_str2id["Ġ"] = i
    for ch in chars:
        if ch not in vocab_str2id:
            i = len(vocab_id2str)
            vocab_id2str[i] = ch
            vocab_str2id[ch] = i

    print(f"Initial vocab (specials + chars): {len(vocab_id2str):,}")

    # Pre-seed common English bigrams — free merges without competing in training loop.
    preseed_merges = []
    for bigram in ENGLISH_PRESEED:
        if len(bigram) == 2 and bigram[0] in vocab_str2id and bigram[1] in vocab_str2id:
            if bigram not in vocab_str2id:
                a_id = vocab_str2id[bigram[0]]
                b_id = vocab_str2id[bigram[1]]
                new_id = len(vocab_id2str)
                vocab_id2str[new_id] = bigram
                vocab_str2id[bigram] = new_id
                preseed_merges.append((a_id, b_id))
    print(f"Pre-seeded {len(preseed_merges)} English bigrams")

    # Pre-seed top Telugu character bigrams computed from the Telugu article text.
    te_text = texts["Telugu"]
    te_pair_counts: Counter = Counter()
    for word in te_text.split():
        for a, b in zip(word, word[1:]):
            if a in vocab_str2id and b in vocab_str2id:
                te_pair_counts[(a, b)] += 1
    te_top = [a + b for (a, b), _ in te_pair_counts.most_common(TELUGU_PRESEED_N * 3)
              if (a + b) not in vocab_str2id][:TELUGU_PRESEED_N]
    for bigram in te_top:
        if bigram[0] in vocab_str2id and bigram[1] in vocab_str2id and bigram not in vocab_str2id:
            a_id = vocab_str2id[bigram[0]]
            b_id = vocab_str2id[bigram[1]]
            new_id = len(vocab_id2str)
            vocab_id2str[new_id] = bigram
            vocab_str2id[bigram] = new_id
            preseed_merges.append((a_id, b_id))
    print(f"Pre-seeded {len(te_top)} Telugu bigrams: {te_top}")

    word_freqs = build_word_freq_dict(texts, weights, vocab_str2id)

    # Apply pre-seeded merges to word_freqs
    for pair in preseed_merges:
        new_id = vocab_str2id[vocab_id2str[pair[0]] + vocab_id2str[pair[1]]]
        word_freqs = merge_word_freqs(word_freqs, pair, new_id)

    num_merges = vocab_size - len(vocab_id2str)
    print(f"Merge rules to learn: {num_merges:,}\n")

    merges = list(preseed_merges)   # pre-seeded merges come first

    for step in range(num_merges):
        if step % 500 == 0:
            print(f"  step {step:>5}/{num_merges}  vocab={len(vocab_id2str):,}", flush=True)

        pairs = get_pairs(word_freqs)
        if not pairs:
            print("No more pairs — stopping early.")
            break

        # Tie-break: prefer pairs with lower token IDs (ASCII/English chars have
        # lower Unicode → lower IDs → get preference on exact frequency ties).
        best_pair = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))

        new_id = len(vocab_id2str)
        word_freqs = merge_word_freqs(word_freqs, best_pair, new_id)
        merges.append(best_pair)

        merged_str = vocab_id2str[best_pair[0]] + vocab_id2str[best_pair[1]]
        vocab_id2str[new_id] = merged_str
        vocab_str2id[merged_str] = new_id

    print(f"\nFinal vocab size: {len(vocab_id2str):,}")
    return vocab_id2str, vocab_str2id, merges


# ── inference ─────────────────────────────────────────────────────────────────

def tokenize(text, merges, vocab_str2id, vocab_id2str):
    """
    Clean BPE inference.  Apply merges in rank order inside each pre-token.
    Uses integer IDs throughout; resolves merged token via vocab_id2str.
    """
    # merge index: (id_a, id_b) → new_id (same order as training)
    # During training, merge step `step` assigned new_id = initial_vocab_size + step
    # We can recover this from vocab_str2id:
    merge_to_new_id = {}
    for rank, (a, b) in enumerate(merges):
        merged_str = vocab_id2str[a] + vocab_id2str[b]
        merge_to_new_id[(a, b)] = vocab_str2id[merged_str]

    merge_rank = {pair: i for i, pair in enumerate(merges)}

    all_token_strs = []
    for tok in pretokenize_text(text):
        ids = [vocab_str2id.get(ch) for ch in tok]
        ids = [i for i in ids if i is not None]

        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                r = merge_rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            new_id = merge_to_new_id[best_pair]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        all_token_strs.extend(vocab_id2str[i] for i in ids)
    return all_token_strs


def compute_fertility(text, merges, vocab_str2id, vocab_id2str):
    words = text.split()
    if not words:
        return 0, 0, 0.0
    tokens = tokenize(text, merges, vocab_str2id, vocab_id2str)
    return len(words), len(tokens), len(tokens) / len(words)


# ── save / load ───────────────────────────────────────────────────────────────

def save_outputs(vocab_id2str, merges, results, gap, score):
    save_dir = f"{BASE}/tokenizer"
    os.makedirs(save_dir, exist_ok=True)

    # vocab.json: {token_str: id}
    vocab_str2id = {v: k for k, v in vocab_id2str.items()}
    with open(f"{save_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_str2id, f, ensure_ascii=False, indent=2)

    # merges.txt: "str_a str_b" one per line
    with open(f"{save_dir}/merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{vocab_id2str[a]} {vocab_id2str[b]}\n")

    # vocabulary.txt: idx TAB token
    with open(f"{BASE}/vocabulary.txt", "w", encoding="utf-8") as f:
        for idx in range(len(vocab_id2str)):
            f.write(f"{idx}\t{vocab_id2str[idx]}\n")

    # stats.json
    fertilities = [v["fertility"] for v in results.values()]
    stats = {
        "results": results,
        "min_fertility": min(fertilities),
        "max_fertility": max(fertilities),
        "gap": gap,
        "score": score,
        "en_constraint": "PASS" if results["English"]["fertility"] <= 1.6 else "FAIL",
        "weights": WEIGHTS,
    }
    with open(f"{BASE}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nOutputs saved:")
    print(f"  {save_dir}/vocab.json   ({len(vocab_id2str)} tokens)")
    print(f"  {save_dir}/merges.txt   ({len(merges)} rules)")
    print(f"  {BASE}/vocabulary.txt")
    print(f"  {BASE}/stats.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BPE Tokenizer (from scratch — integer-ID + word-freq dict)")
    print("=" * 60)

    texts = {}
    for lang, path in LANG_FILES.items():
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        texts[lang] = unicodedata.normalize("NFC", raw)
        print(f"Loaded {lang}: {len(texts[lang]):,} chars")

    vocab_id2str, vocab_str2id, merges = train_bpe(texts, WEIGHTS, VOCAB_SIZE)

    # Verify vocab size
    assert len(vocab_id2str) == VOCAB_SIZE, \
        f"VOCAB SIZE MISMATCH: {len(vocab_id2str)} != {VOCAB_SIZE}"

    # ── Fertility ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Language':<10} {'Words':>8} {'Tokens':>9} {'Fertility':>11}")
    print("-" * 60)

    results = {}
    for lang, text in texts.items():
        nw, nt, f = compute_fertility(text, merges, vocab_str2id, vocab_id2str)
        results[lang] = {"words": nw, "tokens": nt, "fertility": round(f, 4)}
        print(f"{lang:<10} {nw:>8,} {nt:>9,} {f:>11.4f}")

    fertilities = [v["fertility"] for v in results.values()]
    min_f, max_f = min(fertilities), max(fertilities)
    gap   = max_f - min_f
    score = 1000 / gap if gap > 0 else float("inf")

    print("-" * 60)
    print(f"\nMin fertility : {min_f:.4f}")
    print(f"Max fertility : {max_f:.4f}")
    print(f"Gap           : {gap:.4f}")
    print(f"Score (1000/gap): {score:.2f}")

    en_f = results["English"]["fertility"]
    constraint = "PASS" if en_f <= 1.6 else "FAIL"
    print(f"\nEnglish fertility ≤ 1.6 : {constraint}  (actual: {en_f:.4f})")
    print(f"Vocab size check        : {len(vocab_id2str)} == {VOCAB_SIZE}  OK")
    print("=" * 60)

    # ── Worked samples ────────────────────────────────────────────────────────
    samples = {
        "English": "India, officially the Republic of India, is a country in South Asia.",
        "Hindi":   "भारत एशिया महाद्वीप में स्थित एक देश है।",
        "Telugu":  "భారతదేశం దక్షిణాసియాలో ఉన్న ఒక దేశం.",
        "Tamil":   "இந்தியா தென் ஆசியாவில் உள்ள ஒரு நாடு.",
    }
    print("\nWorked samples:")
    for lang, sent in samples.items():
        toks = tokenize(sent, merges, vocab_str2id, vocab_id2str)
        print(f"\n  [{lang}] {sent}")
        print(f"  → {toks}")

    save_outputs(vocab_id2str, merges, results, gap, score)
    return results, gap, score


if __name__ == "__main__":
    main()
