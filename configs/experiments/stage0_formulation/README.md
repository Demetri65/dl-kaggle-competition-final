# Stage 0 Formulations

This project uses four Stage 0 formulations to compare how different answer-selection strategies behave before broader ablations.

These formulations are intentionally narrow and should match their names exactly. Their purpose is to establish strong parent recipes for later stages.

---

## f02_multimodal_index

**Type:** Restricted index scoring

**Goal:** Predict the correct answer as its numeric index.

**Inputs included:**
- image
- question
- full choices list
- hint
- lecture

**Inputs excluded by default:**
- task
- grade
- subject
- topic
- category
- skill

**Target format:**
- one valid numeric label only
- examples:
  - `0`
  - `1`
  - `2`

**How prediction works:**
- the model is prompted with the image and text context
- scoring is restricted to the valid answer indices for that example
- if the question has 3 choices, only `0`, `1`, and `2` are considered
- the label with the highest score is selected

**Why this formulation exists:**
- it is the cleanest multiple-choice baseline
- it keeps the output space small
- it matches the structure of the task directly

---

## f03_multimodal_letter

**Type:** Restricted letter scoring

**Goal:** Predict the correct answer as its letter label.

**Inputs included:**
- image
- question
- full choices list
- hint
- lecture

**Inputs excluded by default:**
- task
- grade
- subject
- topic
- category
- skill

**Target format:**
- one valid letter label only
- examples:
  - `A`
  - `B`
  - `C`

**How prediction works:**
- the model is prompted with the image and text context
- scoring is restricted to the valid letter labels for that example
- if the question has 4 choices, only `A`, `B`, `C`, and `D` are considered
- the label with the highest score is selected

**Why this formulation exists:**
- it tests whether letter labels work better than numeric labels for this model/tokenizer
- it is otherwise structurally parallel to `f02`

---

## f04_candidate_yes_no

**Type:** Minimal candidate verification

**Goal:** Evaluate one candidate answer at a time and decide whether it is correct.

**Inputs included:**
- image
- question
- single candidate answer

**Inputs excluded:**
- full choices list
- hint
- lecture
- task
- grade
- subject
- topic
- category
- skill

**Target format:**
- `Yes`
- `No`

**How prediction works:**
- for each question, the system creates one prompt per candidate answer
- each prompt asks whether that single candidate is correct
- the model scores `Yes` and `No` for each candidate independently
- the candidate with the strongest normalized yes-vs-no score is selected as the final prediction

**Why this formulation exists:**
- it isolates candidate verification behavior
- it removes extra context so we can test whether the model can judge a candidate directly from the image and question alone

---

## f05_candidate_yes_no_full_context

**Type:** Rich candidate verification

**Goal:** Evaluate one candidate answer at a time using the full available context.

**Inputs included:**
- image
- question
- full choices list
- single candidate answer
- hint
- lecture
- task
- grade
- subject
- topic
- category
- skill

**Inputs excluded:**
- none of the standard Stage 0 context fields are intentionally excluded

**Target format:**
- `Yes`
- `No`

**How prediction works:**
- for each question, the system creates one prompt per candidate answer
- each prompt includes the candidate answer plus the full context for the item
- the model scores `Yes` and `No` for each candidate independently
- the candidate with the strongest normalized yes-vs-no score is selected as the final prediction

**Why this formulation exists:**
- it tests whether candidate verification improves when the model is given richer supporting context
- it is the “maximum context” counterpart to `f04`

---

# Summary of differences

| Formulation | Output style | Full choices shown | Candidate shown | Hint | Lecture | Metadata |
|---|---|---:|---:|---:|---:|---:|
| `f02_multimodal_index` | index | yes | no | yes | yes | no |
| `f03_multimodal_letter` | letter | yes | no | yes | yes | no |
| `f04_candidate_yes_no` | yes/no | no | yes | no | no | no |
| `f05_candidate_yes_no_full_context` | yes/no | yes | yes | yes | yes | yes |

---

# Intended use in Stage 0

Recommended Stage 0 use:

- run `f02_multimodal_index` and `f03_multimodal_letter` as zero-shot comparisons on one T4 notebook
- run `f04_candidate_yes_no` and `f05_candidate_yes_no_full_context` as zero-shot comparisons on another T4 notebook
- train a simple LoRA baseline on `f02_multimodal_index` first
- train `f05_candidate_yes_no_full_context` next only if candidate verification looks competitive

This keeps the early experiment set focused while still testing meaningfully different formulations.