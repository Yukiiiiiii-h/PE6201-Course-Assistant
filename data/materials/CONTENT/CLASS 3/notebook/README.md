# PE6201 · Class 3 — C2 build code

**What it teaches:** turn a messy loan-application form into **strict JSON**, score it with a
**10-case eval harness**, then improve the prompt **one change at a time** and watch the pass-rate climb.
This is Assignment 1, part (iii) in miniature.

## Files

| File | Use |
|---|---|
| `PE6201_Class3_Extract_local.ipynb` / `.py` | **Token-free.** Runs `Qwen2.5-1.5B-Instruct` inside Colab. No key, no credits. GPU runtime recommended. |
| `PE6201_Class3_Extract_api.ipynb` / `.py` | Same task + same harness against a hosted model via **OpenRouter** (same key route as Class 1/2). |

Both notebooks are identical from section 1 onward — only the `generate()` helper differs.

## Running it live (instructor)

1. Open the **API** notebook — it is fast and reliable for a live demo (~40 short calls total).
   Key comes from `MY_PRIVATE_OPENROUTER_KEY` env var / Colab secret, else `getpass` prompt.
2. Run top to bottom. The narrative beats are:
   - **§5** run **v1** and let it fail — *do not skip this*, the failure is the teaching moment
   - **§5** read one failure in detail (raw response printed)
   - **§6** the climb table v1 → v4
   - **§7** the pair-build cell (students add a case)
   - **§8** token + cost tally — the bridge into Capsule 3
3. `FAST_MODE = True` cuts to the first 5 cases if you are short on time.

**Before class:** run it once and export `File → Download → .html` as the pre-run fallback
(Class-1 lesson: assume the room's network may fail).

## The four prompt versions

| Version | The one change | Rung (C2 ladder) | Fixes |
|---|---|---|---|
| v1 | plain instruction | 1 · be clear | — (baseline; usually returns prose) |
| v2 | + strict JSON schema | 4 · structured output | shape failures |
| v3 | + 2 worked examples | 2 · few-shot | format consistency (dates, "k" amounts) |
| v4 | + explicit rules | 1 · be clear, precisely | judgement cases (null, annual salary, co-applicant) |

## The 10 eval cases

3 typical · 4 edge · 3 adversarial — all **synthetic**.

| id | What it tests |
|---|---|
| c01 | clean form, everything present |
| c02 | **blank employer** → must return `null`, not invent a company |
| c03 | date written in words ("3 December 1988") |
| c04 | non-SGD currency (CNY) + slashed date |
| c05 | shorthand amounts ("7.2k", "150k") + DD/MM/YYYY |
| c06 | **distractor** — an existing repayment figure that is not the loan |
| c07 | **income stated annually** → must divide by 12 |
| c08 | noisy internal branch chatter around the real fields |
| c09 | missing date of birth → `null` |
| c10 | **two names** → use the primary applicant only |

L1 assertions: valid JSON · exact key set · `full_name` non-empty string · `date_of_birth` is
`YYYY-MM-DD` or null · `employer` string or null · income/loan are plain numbers ·
currency in the allowed ISO set. Plus per-case expectations.

## For the non-technical half

The notebooks are written so a business student can follow them **without reading any Python**:

- **§0 "Five words you need"** — model, prompt, token, JSON, eval, in one table, plus assertion + pass-rate.
- **§1 explains JSON properly** — prose vs labelled fields, the "form not a letter" analogy, and *why*
  structure is what makes a prompt measurable (this is the conceptual hinge of the whole capsule).
- **"In plain words"** boxes open each technical section (the harness, the prompt versions, the run).
- Every dense code cell is preceded by a one-line "what this does" table or sentence.

## Knobs — the notebooks are theirs to keep

9 `TRY THIS (knob)` markers inline, plus a dedicated **§9 playground** with six ready-to-run experiments:

| | Knob | The question it answers |
|---|---|---|
| A | temperature | why do I get a different answer each time? |
| B | rule ablation | which of my rules is actually earning its keep? |
| C | number of examples | is few-shot worth the tokens? |
| D | self-consistency | can I buy reliability with money? (rung 6, priced in C3) |
| E | change the schema | what happens when the business asks for one more field? |
| F | bring your own document | does this survive contact with *my* documents? |

Knob **E** is the sharpest teaching moment: adding a field makes the existing assertions fail on
"unexpected key" — showing that changing the output shape means changing the tests too.

## Notes

- **Your pass-rates will differ from the slide.** The slide table (4/10 → 10/10) is illustrative;
  the notebook prints the real numbers for whichever model you run. Worth saying aloud in class —
  it is a nicer lesson than a rigged demo.
- The **local model scores lower**, deliberately: the method transfers, the quality differs.
- The pair-build example case contains a planted trap — a birth date in **2031** that passes every
  L1 check, because nothing says a DOB must be in the past. That is the point: *a pass-rate is only
  as good as the eval set behind it.*
- All data is synthetic; the notebook says so, and says why, in section 1.
