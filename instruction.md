# Summary Writing Instructions

> Goal: create useful chapter summaries while keeping copyright risk low.
> These are practical guidelines, **not legal advice**.

## Scope

These rules apply to every chapter summary committed to this public repository. They apply to new summaries and to existing ones that are revised.

The safest default is:

- **Public repo:** short, selective, transformative notes.
- **Private notes:** longer and more detailed, but still not verbatim copies.

---

## 1. Keep public summaries short and selective

For a public chapter summary:

- Aim for **300–700 words** (roughly 1–2 pages).
- Cover **3–7 key ideas**, not every section.
- Prefer depth on a few ideas over complete coverage.
- Add your own explanations, analogies, examples, questions, and connections.
- Link to the book and state that the notes are independent study notes.

A good test: the summary should help someone **remember and think about** the ideas, not replace reading the chapter.

---

## 2. Write from understanding, not from the page

- Read the chapter, then close the source and write from memory.
- Use your own sentence structure and your own examples.
- Do **not** paraphrase sentence by sentence.
- Do **not** just swap synonyms or reorder the author's sentences.
- Do **not** mirror the book's full outline. Use section numbers only as references.
- If a sentence feels too close to the source, rewrite it or remove it.

**Transformative** means adding something new: analysis, comparisons, examples, exercises, criticism, or connections. A shortened version of the chapter is not transformative.

---

## 3. Keep quotation minimal

- At most **one short quote per chapter**.
- Keep quotes **under 25 words**.
- Use quotation marks and cite the chapter/page.
- Prefer explaining the idea in your own words over quoting the wording.
- Do not reproduce epigraphs, long passages, or the author's distinctive phrasing.

---

## 4. Do not reproduce media or long examples

Do **not** copy or closely recreate:

- figures, diagrams, charts, or plots,
- tables or screenshots,
- page images from the PDF,
- long code listings,
- the book's datasets or worked examples.

If a figure matters, describe the idea in your own words and cite the page. Draw your own diagram if needed.

Public-domain text may be quoted, but do not copy the book's annotations, framing, or commentary around it.

---

## 5. Follow the code policy

- Prefer your own minimal code examples.
- If you use code from the official [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) repository, follow its **Apache-2.0** license and include attribution.
- Do not copy code from the book text unless you have verified permission.
- Keep snippets short and essential to the explanation.
- Never commit the source PDF, extracted text, or page images to this repository.

Example attribution block:

```text
Some code examples are adapted from
https://github.com/rasbt/LLMs-from-scratch
Licensed under the Apache License 2.0.
```

---

## 6. Cite the source and disclaim affiliation

Put a short source note at the top of each summary:

```markdown
> My notes on Chapter X of *Build a Large Language Model (From Scratch)*
> by Sebastian Raschka (Manning).
> Source: https://www.manning.com/books/build-a-large-language-model-from-scratch
>
> This is an independent study summary, not affiliated with or endorsed by
> the author or publisher.
```

Do not imply that the author or publisher approves of the notes.

---

## 7. Use a low-risk summary template

```markdown
# Chapter X — Topic

> My notes on Chapter X of *Build a Large Language Model (From Scratch)*.
> Source: [book link]

## The idea in one paragraph

Explain the chapter's central idea in your own words.

## Key ideas

- 3–7 takeaways, each in your own words.

## My example

A new, small example that is not copied from the book.

## What surprised me / what I would question

Original analysis, limitations, or open questions.

## One question to test myself

A question that requires understanding, not memorization.

## Further reading

Links to the book, official code, or related material.
```

---

## 8. Pre-publish checklist

Before committing a summary, check:

- [ ] Is the summary short and selective?
- [ ] Is every sentence in my own words?
- [ ] Did I avoid close paraphrase?
- [ ] Are quotes limited to one short, cited quote?
- [ ] Did I avoid reproducing figures, tables, screenshots, and long code?
- [ ] Did I use my own examples?
- [ ] Is there original analysis or commentary?
- [ ] Is the source cited and the affiliation disclaimer present?
- [ ] Is code properly licensed and attributed?
- [ ] Could this replace buying the book? If yes, cut more.
- [ ] Would I be comfortable showing this to the author or publisher?

---

## 9. If in doubt

- Keep detailed notes private.
- Publish only a short, high-level summary.
- Ask the author or publisher for permission.
- Consult an intellectual-property lawyer.
- If the publisher requests removal, comply promptly.

---

## Note on existing summaries

The existing chapter summaries are more detailed than this policy recommends. Before adding more chapters, review them against the checklist above. In particular, the Chapter 2 summary is long and code-heavy; if the goal is low risk, shorten it or keep the detailed version private.
