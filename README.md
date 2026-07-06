# Multilingual BPE Tokenizer

> **Live demo → [mittalutkarsh.github.io/Tokenizer](https://mittalutkarsh.github.io/Tokenizer/)**

A Byte Pair Encoding (BPE) tokenizer trained **from scratch** — no HuggingFace `tokenizers`, no `sentencepiece`, no external ML libraries — on the Wikipedia "India" article in four languages: English, Hindi, Telugu, and Tamil. Vocabulary size is exactly 10,000. The tokenizer is optimised to minimise per-language fertility gap while keeping English fertility ≤ 1.6.

---

## Results

| Language | Script | Words | Tokens | Fertility | Training Weight |
|----------|--------|------:|-------:|----------:|----------------:|
| English | Latin | 10,121 | 15,452 | **1.5267** | 115× |
| Hindi | Devanagari | 8,078 | 11,900 | **1.4731** | 80× |
| Telugu | Telugu | 2,511 | 4,107 | **1.6356** | 230× |
| Tamil | Tamil | 10,297 | 15,218 | **1.4779** | 130× |

| Metric | Value |
|--------|-------|
| Vocabulary size | **10,000** (exact) |
| Min fertility | 1.4731 (Hindi) |
| Max fertility | 1.6356 (Telugu) |
| Gap | **0.1625** |
| Score (1000 ÷ gap) | **6,154** |
| English ≤ 1.6 constraint | ✅ PASS |

---

## Industry Comparison

Same four texts tokenized with production tokenizers. All numbers are real — computed by running each tokenizer on the actual fetched articles.

| Tokenizer | Vocab | English | Hindi | Telugu | Tamil | Gap | Score |
|-----------|------:|--------:|------:|-------:|------:|----:|------:|
| **Ours (from-scratch BPE)** | 10,000 | 1.53 | 1.47 | 1.64 | 1.48 | **0.163** | **6,154** |
| HuggingFace `bert-base-multilingual` | 119,547 | 1.44 | 1.92 | 3.73 | 3.39 | 2.29 | 437 |
| tiktoken `cl100k_base` (GPT-4) | 100,277 | 1.39 | 5.33 | 13.27 | 12.09 | 11.88 | 84 |
| tiktoken `gpt2` (GPT-2) | 50,257 | 1.37 | 8.20 | 20.57 | 24.22 | 22.84 | 44 |

**Key insight:** GPT-2 and GPT-4 were trained on English-dominant web text — Indian scripts fall back to byte-level encoding, producing 13–20× more tokens per word. BERT-multilingual covers 104 languages but allocates only ~1,100 vocab entries per South Asian script. Our domain-specific 10k tokenizer achieves a **14× better score than BERT-multilingual** and a **73× better score than GPT-4** on this language mix — using 12× fewer vocabulary slots.

---

## Repository Structure

```
Tokenizer/
├── index.html              # Interactive widget (open in any browser)
├── train_tokenizer.py      # BPE training — pure Python, no ML libraries
├── india_en.md             # Wikipedia "India" article — English
├── india_hi.md             # Wikipedia "India" article — Hindi
├── india_te.md             # Wikipedia "India" article — Telugu
├── india_ta.md             # Wikipedia "India" article — Tamil
├── tokenizer/
│   ├── vocab.json          # {token: id} — 10,000 entries
│   └── merges.txt          # BPE merge rules, one per line ("a b")
├── vocabulary.txt          # id TAB token, all 10,000 entries
└── stats.json              # Fertility results + training metadata
```

---

## How to Use

### Interactive widget

Open `index.html` in any browser — no server, no installation needed.

- **Vocabulary browser** — search all 10,000 tokens by substring; IDs 0–4 are special tokens, 5–306 are single characters, 307+ are learned BPE merges.
- **Live tokenizer** — type or paste text in any of the four scripts; each token is highlighted in a rotating colour with live fertility readout.
- **On-screen keyboards** — click the language buttons (English / Hindi / Telugu / Tamil) to open a virtual keyboard with vowels, consonants, and matras.

### Programmatic use

```python
import json, unicodedata

# Load vocabulary and merge rules
vocab  = json.load(open("tokenizer/vocab.json"))   # {token: id}
inv    = {v: k for k, v in vocab.items()}           # {id: token}
merges = [tuple(line.split())
          for line in open("tokenizer/merges.txt")] # [(a, b), ...]

def tokenize(text, merges, inv):
    merge_rank = {pair: i for i, pair in enumerate(merges)}
    tokens = []
    for word in unicodedata.normalize("NFC", text).split():
        syms = list(word)
        while len(syms) >= 2:
            best_rank, best_i = None, None
            for i in range(len(syms) - 1):
                r = merge_rank.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        tokens.extend(syms)
    return tokens

# Examples
print(tokenize("India is a country", merges, inv))
# ['India', 'is', 'a', 'country']

print(tokenize("भारत एक देश है", merges, inv))
# ['भारत', 'एक', 'देश', 'है']
```

---

## BPE Algorithm — From Scratch

The tokenizer is implemented in pure Python using only `re`, `collections`, `json`, and `unicodedata`. No external ML libraries.

### Training pipeline (`train_tokenizer.py`)

```
1. Fetch Wikipedia "India" articles (4 languages) → saved as .md files
2. Unicode NFC normalization
3. Build weighted word-frequency dict
   {word_as_char_tuple: frequency × language_weight}
4. Pre-seed vocabulary with high-frequency bigrams (English + Telugu)
5. BPE training loop (9,678 iterations):
   a. Count all adjacent (id_a, id_b) pairs weighted by word frequency
   b. Select the most frequent pair (tie-break: lower token ID wins)
   c. Merge that pair in-place across all words (deque scan, no regex)
   d. Add merged token to vocabulary
6. Save vocab.json, merges.txt, vocabulary.txt, stats.json
```

### Implementation: v1 vs v2

The project went through two BPE implementations.

#### v1 — String / Regex approach

Words stored as space-joined character strings (`"h e l l o"`). Merging used `re.sub()` with lookbehind/lookahead anchors.

```python
# Merge step — v1
bigram  = re.escape(" ".join(pair))
pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
new_word = pattern.sub("".join(pair), word)
```

#### v2 — Integer-ID + Deque (current)

Words stored as tuples of integer token IDs. Pair counting uses `zip`. Merging uses in-place `splice` — no new arrays, no regex.

```python
# Merge step — v2
dq = deque(ids)
while dq:
    cur = dq.popleft()
    if cur == a and dq and dq[0] == b:
        result.append(new_id)
        dq.popleft()
    else:
        result.append(cur)
```

| Dimension | v1 | v2 |
|-----------|----|----|
| Word storage | Space-joined string | Tuple of int IDs |
| Merge engine | `re.compile + re.sub` | Linear `deque` scan |
| Corpus dedup | Full repeated text | Word-frequency dict (12,710 unique types) |
| Fast-skip | None | `if a not in ids: continue` |
| Tie-breaking | Lex-max of string repr (Telugu wins — higher Unicode) | Lower token ID (ASCII/English wins) |

---

## Training Parameters

### Per-language corpus weights

Each language's article is repeated `weight` times in the training corpus, biasing BPE toward learning merges for that script.

```python
WEIGHTS = {
    "English": 115,   # 20.5% of training corpus
    "Hindi":    80,   # 14.3%
    "Telugu":  230,   # 41.1%  ← largest article is only 20 KB; needs most dedicated slots
    "Tamil":   130,   # 23.2%
}
```

Telugu receives 41% of the training budget despite being the shortest article (20 KB vs 65 KB English) because the Telugu script has the richest unique character combinations and requires the most dedicated BPE merge slots.

### Vocabulary pre-seeding

Before the BPE training loop, fixed high-frequency bigrams are injected directly into the vocabulary and applied to the word-frequency dict — "free" merges that don't compete with other languages for budget.

**English pre-seeds (10, hardcoded):**
```
th  he  in  er  re  on  an  en  at  or
```

**Telugu pre-seeds (10, computed at runtime from the Telugu article):**
```
్ర  ార  ాల  లు  ర్  లో  స్  ్య  భా  రా
```

These 20 pre-seeds were the key to satisfying the English ≤ 1.6 fertility constraint without triggering the "merge cascade" effect (see Tuning History below).

### Other training decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Normalization | Unicode NFC | Ensures precomposed and decomposed forms are treated identically |
| Pre-tokenization | Plain whitespace split | Aligns token boundaries with the fertility definition; avoids Ġ-prefix overhead |
| Min frequency | 1 (no filtering) | With weights ≥ 80, even once-seen words have effective frequency ≥ 80 |
| Special tokens | [UNK] [CLS] [SEP] [PAD] [MASK] | Occupy IDs 0–4, count toward 10,000 budget |
| Tie-breaking | Lower token ID wins | ASCII chars have lower Unicode → lower IDs → English gets edge on ties |

**Exact vocab breakdown:**
`302 chars + 5 specials + 10 English pre-seeds + 10 Telugu pre-seeds + 9,673 BPE merges = 10,000`

---

## Tuning History — The Cliff Effect

The most striking finding during optimisation: a difference of **1 unit** in the English weight (110 → 120 out of 230 Telugu) caused English fertility to drop from 1.60 to 1.39, while Telugu jumped from 1.48 to 1.90.

This "cliff effect" is fundamental to greedy BPE — once a critical English character pair crosses the frequency threshold, it cascades: that merge enables downstream merges, all firing in the same training run.

| Configuration | En wt | Te wt | English | Telugu | Gap | Score | Pass? |
|--------------|------:|------:|--------:|-------:|----:|------:|-------|
| HuggingFace baseline | 5 | 10 | 1.566 | 1.562 | 0.157 | 6,383 | ✅ |
| v2 initial (no pre-seeds) | 110 | 230 | 1.603 | 1.481 | 0.126 | 7,924 | ❌ |
| Cliff — En +1 unit | 120 | 230 | 1.395 | 1.901 | 0.507 | 1,974 | ✅ |
| + 10 English pre-seeds | 115 | 230 | 1.527 | 1.636 | 0.163 | 6,124 | ✅ |
| **+ 10 Telugu pre-seeds (final)** | **115** | **230** | **1.527** | **1.636** | **0.163** | **6,154** | **✅** |

Pre-seeding was the breakthrough: injecting common English and Telugu bigrams as free vocabulary entries allowed each language to receive a fertility head-start without competing for BPE budget.

---

## Setup

```bash
# Clone
git clone https://github.com/mittalutkarsh/Tokenizer.git
cd Tokenizer

# Install dependencies (only needed to re-train)
pip install requests

# Re-fetch articles and re-train (takes ~3 minutes on a laptop)
python train_tokenizer.py

# Open the widget
open index.html        # macOS
start index.html       # Windows
xdg-open index.html    # Linux
```

No GPU, no CUDA, no ML framework required. The training script uses only Python standard library plus `requests` for fetching the Wikipedia articles.

---

## Fertility Definition

```
fertility(language) = total_tokens / total_words
```

where `words` = whitespace-delimited tokens of the raw article text, and `tokens` = the output of the BPE tokenizer on the same text. A fertility of 1.0 means every word is a single token; higher values mean more splitting.

**Score** = `1000 ÷ (max_fertility − min_fertility)`. Higher score = narrower fertility gap across languages = more balanced multilingual tokenization.
