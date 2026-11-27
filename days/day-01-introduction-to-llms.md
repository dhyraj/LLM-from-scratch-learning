# Day 1 — Introduction to LLMs

> Independent learning notes. This is not a chapter summary; it is my own path through the ideas.
> For deeper reading: *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning).
> Not affiliated with or endorsed by the author or publisher.

## The question

What is a large language model, and why does predicting the next token produce something that feels like understanding?

## The short answer

An LLM is a neural network trained on a large text corpus to estimate the probability of the next token given the tokens before it. GPT-style models are **decoder-only transformers**: they generate one token at a time and feed each new token back into the model.

Training usually has two stages:

1. **Pretraining** on raw, unlabeled text. The model learns general language patterns by next-token prediction. The result is a **foundation model**.
2. **Finetuning** on labeled or instruction data. The model is adapted for a task such as classification, summarization, translation, or chat.

The transformer architecture comes from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). GPT-3 is a decoder-only model with 175B parameters ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)).

## A real next-token example

Here is a tiny corpus:

```text
the cat sat on the mat . the cat ate .
```

A trigram model can learn what follows the context `"the cat"`:

```python
from collections import Counter

tokens = "the cat sat on the mat . the cat ate .".split()
trigrams = list(zip(tokens, tokens[1:], tokens[2:]))

next_words = Counter(
    c for a, b, c in trigrams if (a, b) == ("the", "cat")
)
print(next_words)
# Counter({'sat': 1, 'ate': 1})
```

After `"the cat"`, the model has seen `"sat"` once and `"ate"` once. Normalize those counts and you get a 50/50 distribution.

This is not an LLM. It is the same objective in miniature. An LLM uses a transformer, a much larger vocabulary, and a much longer context. GPT-3 was trained on roughly 300 billion tokens ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)).

## Three architecture families

| Family | Examples | Training signal | Typical strength |
|---|---|---|---|
| **Decoder-only** | GPT-3, GPT-4, LLaMA | Predict the next token | Generation, chat |
| **Encoder-only** | BERT | Fill in masked tokens | Classification, retrieval |
| **Encoder–decoder** | T5 | Map one sequence to another | Translation, summarization |

- GPT-3: [Brown et al., 2020](https://arxiv.org/abs/2005.14165)
- GPT-4: [OpenAI, 2023](https://arxiv.org/abs/2303.08774)
- LLaMA: [Touvron et al., 2023](https://arxiv.org/abs/2302.13971)
- BERT: [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
- T5: [Raffel et al., 2020](https://arxiv.org/abs/1910.10683)

## What “large” means

“Large” is a combination of three things:

- **Parameters** — the trainable weights. GPT-3 has 175B; the first LLaMA models ranged from 7B to 65B.
- **Data** — GPT-3 was trained on about 300B tokens.
- **Compute** — training runs use large clusters for days or weeks.

A model can have many parameters but still be undertrained. Data and compute matter too.

## Does scale create new abilities?

[Wei et al. (2022)](https://arxiv.org/abs/2206.07682) documented **emergent abilities**: tasks that small models fail but large models can perform. Examples include arithmetic and multi-step reasoning.

That story is not settled. [Schaeffer et al. (2023)](https://arxiv.org/abs/2304.15004) argue that some apparent emergence can be an artifact of the metric: a task can look like a sudden jump when the scoring rule changes. The practical lesson is that scale helps, but claims about “emergence” should be checked carefully.

## What I would build first

Start with a bigram or trigram model, then replace it with a tiny transformer. The rest of this path builds the pieces:

1. text → tokens → token IDs,
2. training windows,
3. token and positional embeddings,
4. attention,
5. a training loop.

## Try it yourself

Change the corpus or the context length in the trigram example. What happens to the distribution? Then sample a sentence from the counts instead of always taking the most likely word.

## Further reading

- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Brown et al., 2020 — [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- Devlin et al., 2019 — [BERT](https://arxiv.org/abs/1810.04805)
- Raffel et al., 2020 — [T5](https://arxiv.org/abs/1910.10683)
- Touvron et al., 2023 — [LLaMA](https://arxiv.org/abs/2302.13971)
- Wei et al., 2022 — [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
- Schaeffer et al., 2023 — [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)
- Book: [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
