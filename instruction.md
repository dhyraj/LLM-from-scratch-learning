# Learning Note Instructions

> Goal: create a low-risk, original learning path while using the book as one reference.
> These are practical guidelines, **not legal advice**.

## Scope

These rules apply to every public note in this repository. The repo is organized by **days and projects**, not by the book's chapters.

The safest default:

- **Public notes:** short, selective, original, and transformative.
- **Private notes:** longer and more detailed, but still not verbatim copies.

---

## 1. Follow our own flow, not the book's outline

- Organize notes around questions, experiments, and projects — not the book's table of contents.
- Do **not** map Day N 1:1 to Chapter N.
- Use the book as **further reading**, not as the skeleton.
- Do not reproduce the book's section order, running examples, figures, tables, or code.
- A shortened chapter is not transformative.

A good test: the note should still make sense if the book did not exist. The book is one source, not the script.

---

## 2. Write from understanding, not from the page

- Read the source, then close it and write from memory.
- Use your own sentence structure, analogies, and examples.
- Do **not** paraphrase sentence by sentence.
- Do **not** just swap synonyms or reorder the author's sentences.
- If a sentence feels too close to the source, rewrite it or remove it.

**Transformative** means adding something new: analysis, comparisons, experiments, exercises, criticism, or connections.

---

## 3. Keep public notes short and selective

- Aim for **300–900 words** per day.
- Cover **3–7 key ideas**, not every section.
- Prefer depth on a few ideas over complete coverage.
- Add your own explanations, examples, questions, and connections.
- Link to the book and state that the notes are independent.

Technical days may need more code or examples, but avoid turning the note into a condensed replacement for a chapter.

---

## 4. Use real, reproducible examples

- Prefer actual outputs from a named tool or model. State the tokenizer/model and version.
- If you show a token ID sequence, generate it with the named tokenizer.
- If you show code output, run the code and paste the real output.
- Do not invent model outputs, benchmark numbers, or token IDs.
- Use public-domain, synthetic, or your own examples. Do not use the book's running examples as your backbone.
- Keep examples small enough that a reader can rerun them.

Example of a good real example:

> `tiktoken.get_encoding("gpt2").encode("unbelievable")` returns `[403, 6667, 11203, 540]`, which decodes to `["un", "bel", "iev", "able"]`.

---

## 5. Cite public research and check every URL

Cite primary sources:

- papers: arXiv abstract pages, ACL Anthology, JMLR, or official DOI pages;
- software: official repositories or documentation;
- model details: the model paper, model card, or official docs.

For each citation, include the title, authors, year, and a stable URL.

Before publishing, open every URL and check:

- it returns HTTP 200,
- the page title matches the cited work,
- it is not a paywall, broken PDF, or search-results page,
- the URL is stable (avoid shorteners and tracking links).

Do not cite a URL you have not checked. If a link is broken or flaky, replace it or remove the claim. Cite specific numbers and non-obvious claims; common knowledge does not need a citation.

---

## 6. Keep quotation minimal

- At most **one short quote per day**.
- Keep quotes **under 25 words**.
- Use quotation marks and cite the source.
- Prefer explaining the idea in your own words.
- Do not reproduce long passages, epigraphs, or distinctive phrasing.

---

## 7. Do not reproduce media or long examples

Do **not** copy or closely recreate:

- figures, diagrams, charts, or plots,
- tables or screenshots,
- page images from the PDF,
- long code listings,
- the book's datasets or worked examples.

If a figure matters, describe the idea in your own words and cite the page. Draw your own diagram if needed. For original diagrams, use a fenced `mermaid` block; keep the syntax simple and GitHub-compatible, and do not recreate the book's figures.

---

## 8. Follow the code policy

- Prefer your own minimal code examples.
- If you use code from the official [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) repository, follow its **Apache-2.0** license and include attribution.
- Do not copy code from the book text unless you have verified permission.
- Keep snippets short and essential.
- Never commit the source PDF, extracted text, or page images to this repository.

Example attribution block:

```text
Some code examples are adapted from
https://github.com/rasbt/LLMs-from-scratch
Licensed under the Apache License 2.0.
```

---

## 9. Use the day template

```markdown
# Day N — Topic

> Independent learning notes. This is not a chapter summary; it is my own path through the ideas.
> For deeper reading: *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning).
> Not affiliated with or endorsed by the author or publisher.

## The question

What am I trying to understand?

## The idea in one paragraph

Explain it in my own words.

## A real example

A reproducible example with actual output.

## What surprised me / limits

Original analysis, limitations, or open questions.

## Further reading

- Paper or official documentation links.
- Book link for deeper reading.
```

---

## 10. Pre-publish checklist

Before committing a note, check:

- [ ] Does this follow my own day/project flow rather than the book's chapter order?
- [ ] Is it 300–900 words and selective?
- [ ] Is every sentence in my own words?
- [ ] Are the examples real, reproducible, and not copied from the book?
- [ ] Did I avoid reproducing figures, tables, screenshots, and long code?
- [ ] Are quotes limited to one short, cited quote?
- [ ] Are the paper URLs valid and primary sources?
- [ ] Are the citations accurate — title, authors, year?
- [ ] Is code properly licensed and attributed?
- [ ] Is the source cited and the affiliation disclaimer present?
- [ ] Could this replace the book? If yes, cut more.
- [ ] Would I be comfortable showing this to the author or publisher?

---

## 11. If in doubt

- Keep detailed notes private.
- Publish only a short, high-level note.
- Ask the author or publisher for permission.
- Consult an intellectual-property lawyer.
- If the publisher requests removal, comply promptly.

---

## Note on the old chapter drafts

The earlier chapter-based summaries were retired from the current tree because they were too detailed and closely followed the book. They may still exist in git history. If the repository is public and that matters, make the repository private or rewrite history. Do not reintroduce chapter-by-chapter summaries.
