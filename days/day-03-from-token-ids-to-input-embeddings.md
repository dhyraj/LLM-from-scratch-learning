# Day 3 — From Token IDs to Input Embeddings

> Independent learning notes. This is not a chapter summary; it is my own path through the ideas.
> For deeper reading: *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning).
> Not affiliated with or endorsed by the author or publisher.

## The question

How do integer token IDs become vectors that carry meaning, and how does the model know the order of the tokens?

## 1. Make input–target pairs

Take a real GPT-2 tokenization:

```text
"The cat sat on the mat."
→ [464, 3797, 3332, 319, 262, 2603, 13]
```

With a window length of 4 and a stride of 1, the input–target pairs are:

| Window | Input IDs | Target IDs |
|---|---|---|
| 1 | `[464, 3797, 3332, 319]` | `[3797, 3332, 319, 262]` |
| 2 | `[3797, 3332, 319, 262]` | `[3332, 319, 262, 2603]` |
| 3 | `[3332, 319, 262, 2603]` | `[319, 262, 2603, 13]` |

The target is the input shifted one position to the right. At each step, the model sees the input window and must predict the next token **without looking ahead**. This is causal language modeling, the objective used by GPT-style models.

A larger stride creates less overlap. With `stride = 4`, the windows would not overlap at all.

## 2. Turn IDs into vectors

A token ID is an index, not a meaning. The embedding layer stores a matrix with one row per vocabulary item:

```text
embedding matrix: (vocabulary_size, embedding_dimension)
vector for token i: row i of the matrix
```

GPT-2’s small model uses an embedding dimension of **768**, a vocabulary of **50,257**, and a context length of **1024** ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)). GPT-3’s 175B model uses an embedding dimension of **12,288** ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)).

Here is the lookup operation in plain Python:

```python
embedding_matrix = [
    [0.10, 0.20, 0.30],  # token 0
    [0.40, 0.50, 0.60],  # token 1
    [0.70, 0.80, 0.90],  # token 2
]

ids = [2, 0, 1]
vectors = [embedding_matrix[i] for i in ids]
print(vectors)
# [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
```

PyTorch’s `torch.nn.Embedding` is the same idea, with trainable weights. During pretraining, the model learns the matrix through backpropagation.

The idea of learning dense word vectors goes back to neural language models and word embeddings, including [Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) and [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781).

A token embedding is **static**: the same token gets the same vector wherever it appears. The transformer’s attention layers later mix in context and produce **contextual** representations.

## 3. Add position information

Self-attention has no built-in sense of order: without position information, it sees a set of tokens, not a sequence. That makes these two sentences look the same at the attention layer:

```text
dog bites man
man bites dog
```

The original Transformer solved this with fixed sinusoidal **positional encodings** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). GPT models use **learned absolute positional embeddings** that are optimized during training ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).

The input embedding is the sum of the token embedding and the position embedding:

```text
input_embedding[i] = token_embedding[i] + position_embedding[i]
```

A small example:

```python
token_vectors = [
    [0.2, -0.1, 0.4],
    [0.5, 0.3, -0.2],
]
position_vectors = [
    [0.0, 0.1, 0.0],
    [0.1, 0.2, 0.1],
]

input_vectors = [
    [t + p for t, p in zip(token, position)]
    for token, position in zip(token_vectors, position_vectors)
]
print(input_vectors)
# [[0.2, 0.0, 0.4], [0.6, 0.5, -0.1]]
```

For a batch, the shapes are:

```text
token IDs:          (batch_size, sequence_length)
token embeddings:   (batch_size, sequence_length, embedding_dimension)
position embeddings:(sequence_length, embedding_dimension)
input embeddings:   (batch_size, sequence_length, embedding_dimension)
```

The position table has one row per position, up to the model’s `context_length`. GPT-2 uses 1024 positions.

### Absolute vs. relative positions

- **Absolute** positions encode the exact index: 0, 1, 2, ...
- **Relative** positions encode the distance between tokens. They can generalize better to sequence lengths the model has not seen ([Shaw et al., 2018](https://arxiv.org/abs/1803.02155)).

Many modern LLMs use rotary position embeddings (RoPE), a relative scheme introduced in [Su et al., 2021](https://arxiv.org/abs/2104.09864). The GPT-2 style implementation uses learned absolute positions instead.

## The full input pipeline

```text
text
  → tokenizer
  → token IDs
  → input/target windows
  → token embeddings
  → + positional embeddings
  → transformer
```

## Try it yourself

1. Tokenize a sentence with `tiktoken` and build input–target pairs with `max_length=4`, `stride=1`.
2. Implement the embedding lookup with a small matrix.
3. Add a position vector to each token vector and check the shapes.
4. Explain why the target must be shifted one position to the right.

## Further reading

- Bengio et al., 2003 — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Mikolov et al., 2013 — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al., 2019 — [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Shaw et al., 2018 — [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- Su et al., 2021 — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Book: [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
