# Chapter 2 — Working with Text Data

> Summary of **Chapter 2: Working with Text Data** from *Build a Large Language Model (From Scratch)*.
> Based on the notes PDF `Building LLMs From Scratch (1).pdf`, pp. 7–21.

---

## The big picture

An LLM cannot consume raw text. Chapter 2 builds the data pipeline that turns text into the numerical inputs a GPT-style transformer expects:

```text
raw text
   │  tokenize
   ▼
tokens
   │  map to integers
   ▼
token IDs
   │  sliding window
   ▼
(input, target) token-ID pairs
   │  embedding layer
   ▼
token embeddings
   │  + positional embeddings
   ▼
input embeddings → transformer
```

This is **Stage 1: data preparation and sampling**. The output of this chapter is not a trained model; it is a repeatable pipeline that turns a text corpus into batches of token IDs and then into the input embeddings consumed by the transformer.

Two ideas are easy to confuse:

- **Tokenization** — text → discrete tokens.
- **Embedding** — token IDs → continuous vectors.

---

## 2.1 Understanding word embeddings

Neural networks operate on numbers, not strings. An **embedding** maps discrete objects — words, images, audio — into a continuous vector space:

```text
"cat"  →  [ 0.21, -1.04, 0.77, ... ]
```

### Types of text embeddings

- **Word embeddings** — the most common form of text embedding; each word/token gets a vector.
- **Sentence, paragraph, or document embeddings** — a whole span of text gets one vector. These are popular for **retrieval-augmented generation (RAG)**.

Word embeddings can have anywhere from one to thousands of dimensions. Two-dimensional embeddings are useful for visualization: the notes show a plot where related words such as birds, “squirrel”, “London”, and “England” cluster in meaningful ways.

At this stage, the important point is that the embedding layer is **trainable**. The vectors are not hand-designed; the model learns them during pretraining.

> **Static vs. contextual embeddings:** The token embedding layer gives each token a position-independent vector. Later transformer layers produce **contextualized** representations that depend on the surrounding tokens.

---

## 2.2 Tokenizing text

**Tokenization** splits raw text into smaller units called **tokens**. A token can be a word, punctuation mark, subword, or character.

A simple word-level tokenizer can use a regular expression to separate words and punctuation:

```python
import re

text = "The quick brown fox jumps over the lazy dog."

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(preprocessed)
# ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
```

The regex splits on:

- whitespace (`\s`),
- common punctuation marks,
- double hyphens (`--`).

Whitespace-only pieces are removed, but punctuation is kept as its own token.

### Building a vocabulary

The **vocabulary** is the set of all unique tokens the tokenizer knows about. A common approach is to sort the unique tokens alphabetically and assign each one an integer:

```python
all_words = sorted(set(preprocessed))
vocab = {token: idx for idx, token in enumerate(all_words)}

print(vocab)
# {'!': 0, ',': 1, '.': 2, 'The': 3, ...}
```

The integer assigned to a token is its **token ID**. The IDs are arbitrary — the number itself has no meaning.

### Limitation of a word-level tokenizer

A fixed word-level vocabulary cannot handle words it has never seen. Encoding a new word would raise a `KeyError`. This motivates:

- a special **unknown token** (`<|unk|>`), and
- **subword tokenization** such as byte pair encoding.

The book uses a short story (“The Verdict”) as its running text sample.

---

## 2.3 Converting tokens into token IDs

Token IDs let the model work with integers. The two core operations are:

- **encode** — text → tokens → token IDs.
- **decode** — token IDs → tokens → text.

A minimal tokenizer class looks like this:

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove the extra space before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

### Why decoding needs post-processing

`decode` joins tokens with spaces. Without cleanup, punctuation would look wrong:

```text
"Hello , world !"   →   "Hello, world!"
```

The regex removes the space before punctuation.

### The pipeline

The notes show the complete path:

```text
input text
   → tokenized text
   → token IDs
   → token embeddings
   → GPT-like decoder-only transformer
   → postprocessing
   → output text
```

Token IDs are not meaningful numbers; they are simply indices into the vocabulary.

---

## 2.4 Adding special context tokens

Some information is not present in the raw text but is still useful to the model. **Special tokens** mark this extra context.

- `<|unk|>` — unknown word/token.
- `<|endoftext|>` — boundary between unrelated texts, or end of a document.
- `[BOS]` — beginning of a sequence.
- `[EOS]` — end of a sequence.
- `[PAD]` — padding to a fixed length.

### Why `<|endoftext|>` matters

Suppose we concatenate several independent documents into one training stream. Without a boundary marker, the model may learn spurious relationships between the end of one document and the start of the next.

The notes show `<|endoftext|>` being inserted between independent text sources:

```text
... the underdog team finally clinched the championship ...
<|endoftext|>
... Elara and Finn lived with kindness and wisdom ...
<|endoftext|>
... the Dow Jones Industrial Average closed up 250 points ...
```

GPT-2 uses `<|endoftext|>` as its end-of-text token.

### Adding special tokens to the vocabulary

Special tokens must be added explicitly; they are not learned from raw text:

```python
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: idx for idx, token in enumerate(all_tokens)}
```

An updated tokenizer can map unknown words to `<|unk|>`:

```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

Example of joining two texts with a document boundary:

```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
```

---

## 2.5 Byte pair encoding

A word-level vocabulary has two problems:

- it can become enormous,
- it cannot represent words it has never seen.

**Byte pair encoding (BPE)** solves this with **subword tokenization**. Instead of storing whole words only, BPE builds a vocabulary of frequent subword units.

### How BPE works

1. Start with individual characters (or bytes).
2. Count the most frequent adjacent pair of symbols in the corpus.
3. Merge that pair into a new symbol.
4. Repeat for a fixed number of merges.
5. The result is a vocabulary plus a set of merge rules.

Common words end up as single tokens; rare or unseen words are split into smaller subwords or characters.

```text
"unbelievable"
   → ["un", "believ", "able"]   (illustrative)
```

### BPE in practice

GPT-2 and GPT-3 use BPE-style tokenization. GPT-2 uses **byte-level BPE**, which can represent any Unicode text as bytes and therefore avoids a hard “unknown token” problem. The `tiktoken` library provides a fast implementation:

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

ids = tokenizer.encode(
    "Hello, world!",
    allowed_special={"<|endoftext|>"}
)
print(ids)

text = tokenizer.decode(ids)
print(text)  # Hello, world!
```

The GPT-2 vocabulary contains **50,257 tokens**.

### Comparing tokenization strategies

| Strategy | Unit | Vocabulary | Unknown words |
|---|---|---|---|
| Word-level | Words and punctuation | Large, fixed | Fails, or uses an unknown token |
| Character-level | Individual characters | Small | No unknown words, but sequences become long |
| Subword / BPE | Frequent subwords | Medium (≈50k for GPT-2) | Split into subwords or characters |

> **Practical note:** GPT-2 BPE tokens often include a leading space. A token that looks like `Ġworld` represents `" world"`. This is why decoded text can look correct even when token boundaries do not match word boundaries.

BPE is reversible for valid token sequences, but token IDs are **tokenizer-specific**. Do not mix IDs from different tokenizers.

---

## 2.6 Data sampling with a sliding window

The pretraining objective is **next-token prediction**: given a sequence of tokens, predict the token that comes next. The model must make each prediction using only the input window; it cannot look ahead at future tokens.

To create training examples, take a window of tokens as the **input** and shift it one position to the right for the **target**:

```text
input:   [The, quick, brown, fox]
target:  [quick, brown, fox, jumps]
```

More generally, for a window starting at position `i`:

```python
input_chunk  = token_ids[i : i + max_length]
target_chunk = token_ids[i + 1 : i + max_length + 1]
```

The notes show these input-target pairs:

```text
and                         → established
and established             → himself
and established himself     → in
and established himself in  → a
```

### Sliding window and stride

A **sliding window** moves through the token stream. The **stride** controls how far the window moves each step:

- `stride = 1` — maximum overlap; windows shift by one token.
- `stride = max_length` — no overlap; windows are adjacent.
- Intermediate values — partial overlap.

Example with the text “In the heart of the city stood the old library...” and `max_length = 4`:

```text
stride = 1:
  batch 1 input: "In the heart of"
  batch 2 input: "the heart of the"

stride = 4:
  batch 1 input: "In the heart of"
  batch 2 input: "the city stood the"
```

### Dataset implementation

```python
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

A helper can wrap the dataset in a PyTorch `DataLoader`:

```python
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
```

Example:

```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False,
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
inputs, targets = first_batch
```

With `N` tokens, window length `L`, and stride `S`, the loop creates

```text
ceil((N - L) / S)
```

windows when `N > L`.

### Batch size trade-off

- **Small batches** use less memory but produce noisier weight updates.
- **Large batches** give more stable updates but require more memory.

Batch size is a hyperparameter to tune during training. `drop_last=True` avoids an incomplete final batch; `shuffle=True` is typical for training.

---

## 2.7 Creating token embeddings

Token IDs are still discrete integers. The **embedding layer** converts each ID into a dense, continuous vector.

In PyTorch, `torch.nn.Embedding` is essentially a lookup table:

```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(6, 3)

print(embedding_layer.weight)
# A 6 × 3 matrix: one 3-dimensional vector per token ID

token_ids = torch.tensor([5, 1, 3, 2])
print(embedding_layer(token_ids))
# A 4 × 3 matrix: one row per input token ID
```

The weight matrix has shape:

```text
(vocabulary_size, embedding_dimension)
```

The lookup returns the row corresponding to each token ID:

```text
token ID 5  →  row 5 of the embedding matrix
token ID 1  →  row 1 of the embedding matrix
token ID 3  →  row 3 of the embedding matrix
token ID 2  →  row 2 of the embedding matrix
```

### Shapes through the model

```text
input token IDs:      (batch_size, sequence_length)
token embeddings:     (batch_size, sequence_length, embedding_dimension)
```

### Key properties

- The embedding weights are **trainable**; they are initialized randomly and learned during pretraining.
- The lookup is equivalent to multiplying a one-hot vector by the embedding matrix, but the lookup is much more efficient.
- GPT-2 uses an embedding dimension of **768** (for the small model); GPT-3 uses **12,288**.
- Token embeddings are **position-independent**. The same token gets the same vector regardless of where it appears.
- The mapping is deterministic: the same token ID always returns the same vector, which helps reproducibility.

> **Static vs. contextual:** The embedding layer gives a static vector per token. The transformer’s attention and feed-forward layers later mix in context, producing contextual representations.

---

## 2.8 Encoding word positions

Token embeddings alone do not encode word order. The self-attention mechanism is also position-agnostic, so without extra information the model would see a sequence as an unordered “bag” of tokens.

To fix this, add **positional embeddings** to the token embeddings:

```text
input embeddings = token embeddings + positional embeddings
```

The addition is element-wise, so both vectors must have the same dimension.

### Absolute vs. relative positional embeddings

| Type | Idea | Strength |
|---|---|---|
| **Absolute** | Give each position (0, 1, 2, ...) its own vector | Simple; used by GPT models |
| **Relative** | Encode the distance between tokens | Better generalization to different sequence lengths |

The notes emphasize:

- **Relative** positional embeddings focus on “how far apart” tokens are rather than “at which exact position” they sit. This can generalize better to sequence lengths the model has not seen.
- **OpenAI’s GPT models use absolute positional embeddings** that are optimized during training — not the fixed sinusoidal encodings used in the original Transformer.

### Creating positional embeddings

```python
context_length = 1024
embedding_dim = 768

pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

pos_ids = torch.arange(seq_len)
pos_embeddings = pos_embedding_layer(pos_ids)

input_embeddings = token_embeddings + pos_embeddings
```

The notes illustrate the addition with a simple example:

```text
token embeddings:       [1, 1, 1]   [1, 1, 1]   [1, 1, 1]   [1, 1, 1]
positional embeddings:  [1, 1, 1]   [2, 2, 2]   [3, 3, 3]   [4, 4, 4]
                        ---------   ---------   ---------   ---------
input embeddings:       [2, 2, 2]   [3, 3, 3]   [4, 4, 4]   [5, 5, 5]
```

### Context length

`context_length` is the maximum input size the model supports. GPT-2 uses **1024** positions. The positional embedding table therefore has one row per position:

```text
(context_length, embedding_dimension)
```

A sequence longer than `context_length` cannot be represented by this table. The positional vectors are learned parameters, just like the token embeddings.

> **Beyond the book:** Many modern LLMs use rotary position embeddings (RoPE) or other relative schemes. The GPT implementation in this book uses learned absolute positional embeddings.

---

## 2.9 Summary

### Key takeaways

- **LLMs need numbers, not raw text.** Embeddings map discrete data into continuous vector spaces.
- **Tokenization** splits text into tokens; **token IDs** are integer indices into a vocabulary.
- **Special tokens** such as `<|unk|>` and `<|endoftext|>` handle unknown words and document boundaries.
- **BPE** is a subword tokenizer that handles unknown words by breaking them into smaller units. GPT-2 and GPT-3 use BPE-style tokenization; GPT-2’s vocabulary has 50,257 tokens.
- A **sliding window** creates input-target pairs for next-token prediction. Targets are the inputs shifted one position to the right.
- **Stride** controls overlap between windows; **batch size** trades memory for update stability.
- An **embedding layer** is a trainable lookup table mapping token IDs to dense vectors.
- **Token embeddings have no sense of position.** Positional embeddings are added to inject order information.
- GPT models use **learned absolute positional embeddings**; relative positional embeddings are an alternative that can generalize better across sequence lengths.

### The complete input pipeline

```text
raw text
   │  tokenizer (BPE)
   ▼
tokens
   │  vocabulary lookup
   ▼
token IDs
   │  sliding window
   ▼
(input IDs, target IDs)
   │  embedding layer
   ▼
token embeddings
   │  + positional embeddings
   ▼
input embeddings
   │
   ▼
GPT-style transformer (Chapters 3–4)
```

At the end of Chapter 2, we have a `DataLoader` that yields batches of token IDs, plus an input-embedding pipeline that produces:

```text
(batch_size, sequence_length, embedding_dimension)
```

These are the inputs the attention mechanism will consume in Chapter 3.

---

## Key terms

| Term | Meaning |
|---|---|
| **Token** | A unit of text: word, punctuation, subword, or character. |
| **Tokenization** | Splitting raw text into tokens. |
| **Vocabulary** | The set of tokens the tokenizer knows, each mapped to an ID. |
| **Token ID** | The integer index assigned to a token. |
| **Special token** | A token added for structure/context, such as an unknown-word token or an end-of-text token. |
| **BPE** | Byte pair encoding; a subword tokenization algorithm. |
| **Subword** | A token smaller than a word but larger than a character. |
| **tiktoken** | OpenAI’s fast BPE tokenizer library. |
| **Context length** | Maximum number of positions/tokens the model supports. |
| **Sliding window** | A moving window that extracts overlapping input-target pairs. |
| **Stride** | How far the window moves at each step. |
| **Batch size** | Number of windows processed together in one training step. |
| **Token embedding** | A learned dense vector for a token ID. |
| **Positional embedding** | A vector added to a token embedding to encode position. |
| **Absolute position** | Encoding the exact index of a token. |
| **Relative position** | Encoding the distance between tokens. |
| **RAG** | Retrieval-augmented generation; often uses sentence/paragraph embeddings. |

---

## Quick self-check

1. Why can’t an LLM process raw text directly?
2. What is the difference between tokenization and embedding?
3. Why do we need `<|unk|>` and `<|endoftext|>`?
4. How does BPE handle a word it has never seen?
5. Given token IDs `[1, 2, 3, 4, 5]`, `max_length = 3`, and `stride = 1`, what are the first two input-target pairs?
6. What does `torch.nn.Embedding` do, and what is the shape of its weight matrix?
7. Why are positional embeddings needed if token embeddings already exist?
8. What is the difference between absolute and relative positional embeddings?
