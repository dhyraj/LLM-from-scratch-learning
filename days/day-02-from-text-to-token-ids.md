# Day 2 — From Text to Token IDs

> Independent learning notes. This is not a chapter summary; it is my own path through the ideas.
> For deeper reading: *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning).
> Not affiliated with or endorsed by the author or publisher.

## The question

How does raw text become a sequence of integers that a model can process?

## The pipeline

```text
raw text
  → tokens
  → token IDs
  → training windows
  → embeddings
```

This day covers the first two steps: splitting text into tokens and mapping those tokens to IDs.

## A real GPT-2 tokenization

GPT-2 uses byte-level byte pair encoding (BPE). The `tiktoken` library exposes the same tokenizer:

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = "The quick brown fox jumps over the lazy dog."

ids = enc.encode(text)
print(ids)
# [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]

tokens = [
    enc.decode_single_token_bytes(i).decode("utf-8")
    for i in ids
]
print(tokens)
# ['The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog', '.']
```

Three things are worth noticing:

- The text has 9 words plus a period, but it becomes **10 tokens** because the period is its own token.
- Many tokens include a **leading space**. The token `' quick'` is not the same as `'quick'`.
- The IDs are indices into GPT-2’s **50,257-token vocabulary**. They are not meaningful numbers by themselves.

The GPT-2 paper is [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). The tokenizer implementation is [openai/tiktoken](https://github.com/openai/tiktoken).

## Why not just use words?

A word-level vocabulary has three problems:

1. **Unknown words** — a new word is not in the vocabulary.
2. **Vocabulary size** — natural language has an enormous number of distinct words.
3. **Morphology** — `"run"`, `"runs"`, `"running"`, and `"runner"` are related, but a word-level tokenizer treats them as unrelated IDs.

Subword tokenization finds a middle ground. Common words stay whole; rare words are split into smaller pieces.

## BPE in one paragraph

BPE starts with small units — characters, or bytes in GPT-2 — and repeatedly merges the most frequent adjacent pair into a new token. After enough merges, the vocabulary contains common words and useful subwords. New text is encoded by applying the learned merges. The method was introduced for neural machine translation by [Sennrich et al., 2016](https://aclanthology.org/P16-1162/).

A real GPT-2 example:

```python
print(enc.encode("unbelievable"))
# [403, 6667, 11203, 540]

print([
    enc.decode_single_token_bytes(i).decode("utf-8")
    for i in enc.encode("unbelievable")
])
# ['un', 'bel', 'iev', 'able']
```

The word is not in the vocabulary as a single unit, but it can still be represented.

## Token IDs are tokenizer-specific

The same word can split differently in different tokenizers:

| Tokenizer | Tokens for `unbelievable` | IDs |
|---|---|---|
| `gpt2` | `un`, `bel`, `iev`, `able` | `[403, 6667, 11203, 540]` |
| `cl100k_base` | `un`, `belie`, `vable` | `[359, 32898, 24694]` |
| `o200k_base` | `un`, `bel`, `ievable` | `[373, 9880, 45794]` |

These outputs were generated with `tiktoken`. The lesson: **never mix token IDs from different tokenizers**. An ID only means something inside the vocabulary that produced it.

GPT-2 uses byte-level BPE and a 50,257-token vocabulary. GPT-3 also uses BPE ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)). Other models use different schemes: BERT uses WordPiece ([Devlin et al., 2019](https://arxiv.org/abs/1810.04805)), and T5 uses SentencePiece ([Raffel et al., 2020](https://arxiv.org/abs/1910.10683)).

## Special tokens

Special tokens mark structure that is not ordinary text. Common examples:

- `<|endoftext|>` — GPT-2’s end-of-text and document-boundary token.
- `<|unk|>` — an unknown-word token used by some word-level tokenizers.
- `[BOS]`, `[EOS]`, `[PAD]` — beginning of sequence, end of sequence, and padding in other model families.

GPT-2’s `<|endoftext|>` is token **50256**:

```python
ids = enc.encode(
    "Hello <|endoftext|> world",
    allowed_special={"<|endoftext|>"}
)
print(ids)
# [15496, 220, 50256, 995]
```

The `allowed_special` argument tells `tiktoken` to treat the marker as a special token instead of raising an error. Special tokens are part of the vocabulary, but they are not ordinary text. Follow the tokenizer’s documentation rather than inventing your own markers.

## Try it yourself

1. Tokenize a sentence you wrote with `gpt2` and `cl100k_base`.
2. Decode both ID sequences and confirm that you get the original text back.
3. Find a word that splits into different pieces in the two tokenizers.
4. Try an emoji or a non-English word and see how byte-level BPE handles it.

## Further reading

- Sennrich et al., 2016 — [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- Radford et al., 2019 — [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Brown et al., 2020 — [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- Devlin et al., 2019 — [BERT](https://arxiv.org/abs/1810.04805)
- Raffel et al., 2020 — [T5](https://arxiv.org/abs/1910.10683)
- [openai/tiktoken](https://github.com/openai/tiktoken)
- Book: [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
