# Chapter 1 — Understanding LLMs

> Summary of **Chapter 1: Understanding LLM** from *Build a Large Language Model (From Scratch)*.
> Based on the notes PDF `Building LLMs From Scratch (1).pdf`, pp. 3–6.

---

## The big picture

A **large language model (LLM)** is a deep neural network trained on a very large text corpus so that it can understand, generate, and respond to human-like text. Building one has two main stages:

1. **Pretraining** — train on raw, unlabeled text to produce a **foundation model**.
2. **Finetuning** — continue training on labeled data to specialize the model for a particular task.

GPT, the model built in this book, is a **decoder-only transformer** that generates text one token at a time.

---

## 1.1 What is an LLM?

### Where LLMs fit in AI

```
Artificial intelligence (AI)
├── Machine learning (ML)
│   └── Deep learning (DL)
│       └── Large language models (LLMs) — built on transformers
└── Other AI approaches
    ├── Rule-based systems
    ├── Genetic algorithms
    ├── Expert systems
    ├── Fuzzy logic
    └── Symbolic reasoning
```

LLMs are a subset of **deep learning**, which is itself a subset of **machine learning** and AI.

### Traditional machine learning vs. deep learning

| | Traditional machine learning | Deep learning |
|---|---|---|
| **Features** | Manually engineered by human experts | Learned automatically from data |
| **Example: spam detection** | Experts define features such as counts of trigger words ("prize", "win", "free"), number of exclamation marks, ALL-CAPS words, and suspicious links | The model learns useful features directly from the raw email text |
| **Data** | Requires labeled data **and** manual feature extraction | Requires data, but no manual feature extraction |

> **Note:** Both approaches still require collecting data (e.g., which emails are spam and which are not). The key difference is that traditional ML additionally requires experts to identify and extract the relevant features.

### What makes an LLM "large"?

"Large" refers to two things:

- **Number of parameters** — the trainable weights in the network, typically in the billions.
- **Size of the training data** — massive text corpora, often trillions of words/tokens.

LLMs are based on the **transformer** architecture. A key component of transformers is the **self-attention mechanism**, which lets the model weigh the importance of different words or tokens in a sequence relative to each other.

### How LLMs are trained

- **Pretraining** uses unlabeled text. The model learns general language patterns, typically by predicting the next token in a sequence.
- **Finetuning** uses labeled data to adapt the pretrained model to specific tasks.

The two most popular categories of finetuning are:

1. **Instruction finetuning** — teach the model to follow instructions (e.g., a chat/personal assistant).
2. **Finetuning for classification tasks** — adapt the model to assign labels (e.g., spam vs. not spam).

---

## 1.2 Applications of LLMs

After **pretraining**, a foundation model has a few basic capabilities:

- **Text completion** — continue a piece of text.
- **Few-shot capabilities** — perform a new task from a handful of examples included in the prompt, without updating the model weights.

After **finetuning**, LLMs can be specialized for tasks such as:

- Classification
- Summarization
- Translation
- Personal assistant / chat
- Other task-specific applications

---

## 1.3 Stages of building and using LLMs

```
Raw, unlabeled text
(Internet texts, books, Wikipedia, research articles — trillions of words)
        │
        │  Pretraining
        ▼
Pretrained LLM / foundation model
  Basic capabilities: text completion, few-shot
        │
        │  Finetuning on a labeled dataset
        ▼
Finetuned LLM
  Classification, summarization, translation, personal assistant, ...
```

The notes break the process into **three stages**:

| Stage | What happens | Output |
|---|---|---|
| **1. Build** | Data preparation and sampling, attention mechanism, LLM architecture | An untrained LLM implementation |
| **2. Pretrain** | Training loop, model evaluation, loading pretrained weights | Foundation model |
| **3. Finetune** | Train on a labeled dataset | Classifier (class labels) or personal assistant (instruction dataset) |

There are two common finetuning routes from the foundation model:

- **Classification finetuning** → a classifier trained on a dataset with class labels.
- **Instruction finetuning** → a personal assistant/chat model trained on an instruction dataset.

---

## 1.4 Using LLMs for different tasks

Different transformer architectures suit different tasks:

| Architecture | Example | Training objective | Typical use |
|---|---|---|---|
| **Encoder–decoder** | T5, BART | Read the full input, then generate the output | Translation, summarization |
| **Encoder-only** | BERT | Masked language modeling: fill in randomly hidden words | Classification, understanding tasks |
| **Decoder-only** | GPT | Predict the next token | Text generation, chat |

### Encoder–decoder: translation example

The notes illustrate machine translation with an encoder–decoder:

1. The input text to be translated is provided.
2. The input is preprocessed for the encoder.
3. The encoder has access to the **complete input text** and produces text encodings.
4. The encoder returns embedding vectors that are passed to the decoder.
5. The decoder receives a partial output (the translation so far).
6. The partial output is preprocessed for the decoder.
7. The decoder generates the translated text **one word at a time**.
8. The complete translated output is produced.

### BERT vs. GPT

| | **BERT** | **GPT** |
|---|---|---|
| **Architecture** | Encoder-only | Decoder-only |
| **Training input** | Text with words randomly masked | Incomplete text |
| **Objective** | Fill in the missing words to reconstruct the original sentence | Learn to generate the next word |
| **Strength** | Understanding/classification | Text generation |

Both rely on **self-attention** to model relationships between tokens.

---

## 1.5 Utilizing large datasets

LLMs require enormous amounts of text. The notes use GPT-3's pretraining mixture as an example:

| Dataset | Description | Tokens | Proportion |
|---|---|---:|---:|
| CommonCrawl (filtered) | Web crawl data | 410 billion | 60% |
| WebText2 | Web crawl data | 19 billion | 22% |
| Books1 | Internet-based book corpus | 12 billion | 8% |
| Books2 | Internet-based book corpus | 55 billion | 8% |
| Wikipedia | High-quality text | 3 billion | 3% |
| **Total** | | **~499 billion** | **~100%** |

Source notes:

- **Wikipedia** — English-language Wikipedia.
- **Books1** — likely a sample from Project Gutenberg.
- **Books2** — likely from Libgen.
- **CommonCrawl** — a filtered subset of the CommonCrawl database.
- **WebText2** — text from web pages linked from Reddit posts with 3+ upvotes.

### Emergent behavior

The ability to perform tasks that the model was **not explicitly trained** to perform is called **emergent behavior**.

Examples include reasoning and arithmetic abilities that appear as model size increases. These abilities are not driven by individual parameters or specific tasks; they are a spontaneous product of increased system scale and complexity.

---

## 1.6 A closer look at the GPT architecture

- **GPT** stands for **Generative Pretrained Transformer**.
- It is a **decoder-only** transformer: it uses the decoder part of the transformer architecture, without a separate encoder.
- Its pretraining objective is **next-token prediction**: given the tokens so far, predict the next one. This is also called **causal** or **autoregressive** language modeling.
- Text is generated **one token at a time**; each newly generated token is fed back into the model as input for the next step.
- Because generation is causal, each token can attend only to **previous** tokens, not future ones.
- GPT-3, the model whose data mixture is shown above, has **175 billion parameters**.

This is the architecture the book implements from scratch in the later chapters.

---

## 1.7 Building a large language model

The three-stage process from the notes maps onto the book's roadmap:

| Stage | Steps | Chapters |
|---|---|---|
| **1. Build the LLM** | Data preparation and sampling; attention mechanism; LLM architecture | 2–4 |
| **2. Pretrain** | Training loop; model evaluation; load pretrained weights | 5 |
| **3. Finetune** | Classification finetuning or instruction finetuning | 6–7 |

More specifically:

1. **Data preparation and sampling** — tokenize text, convert tokens to IDs, create training batches with a sliding window, and build token embeddings.
2. **Attention mechanism** — implement self-attention, causal attention, and multi-head attention.
3. **LLM architecture** — assemble the GPT model with transformer blocks, layer normalization, and feed-forward layers.
4. **Pretraining** — train on unlabeled text, evaluate the loss, and generate text.
5. **Finetuning** — adapt the pretrained model for classification or instruction following.

---

## 1.8 Summary

### Key takeaways

- **LLMs are deep neural networks** trained on massive text corpora to understand and generate human-like text.
- **"Large"** means many parameters and a very large training dataset.
- **Pretraining** is self-supervised learning on unlabeled text; it produces a **foundation model**.
- **Finetuning** is supervised learning on labeled data; it specializes the model for tasks.
- The two main finetuning categories are **instruction finetuning** and **classification finetuning**.
- **GPT is decoder-only** and generative: it predicts the next token and generates text one token at a time.
- **BERT is encoder-only** and uses masked language modeling to fill in missing words.
- **Encoder–decoder** models are well suited to sequence-to-sequence tasks such as translation.
- **Self-attention** is the core mechanism that lets transformers weigh the importance of different tokens.
- **Scale enables emergent behavior** — abilities such as reasoning and arithmetic can appear as models grow, even without explicit training for those tasks.
- Building an LLM follows three stages: **build the architecture → pretrain on unlabeled text → finetune for specific tasks**.

---

## Key terms

| Term | Meaning |
|---|---|
| **LLM** | Large language model; a deep neural network trained on massive text data. |
| **Foundation model** | A pretrained LLM that can be adapted to many downstream tasks. |
| **Pretraining** | Initial training on large-scale, unlabeled text. |
| **Finetuning** | Further training on labeled data for a specific task. |
| **Instruction finetuning** | Finetuning an LLM to follow instructions, as in a chat assistant. |
| **Classification finetuning** | Finetuning an LLM to predict class labels. |
| **Transformer** | The neural network architecture underlying modern LLMs. |
| **Self-attention** | A mechanism that weighs the relevance of different tokens in a sequence to each other. |
| **Encoder** | Transformer component that processes the full input into representations. |
| **Decoder** | Transformer component that generates output, typically one token at a time. |
| **Token** | A unit of text (word or subword) processed by the model. |
| **Causal / autoregressive** | Predicting each next token using only the preceding tokens. |
| **Few-shot learning** | Performing a task from a few examples in the prompt without weight updates. |
| **Emergent behavior** | Capabilities that appear with scale but were not explicitly trained. |

---

## Quick self-check

1. What does the "large" in "large language model" refer to?
2. What is the difference between pretraining and finetuning?
3. Why is GPT described as decoder-only?
4. What is self-attention, and why is it important?
5. What are the two most popular categories of LLM finetuning?
6. What is emergent behavior?
