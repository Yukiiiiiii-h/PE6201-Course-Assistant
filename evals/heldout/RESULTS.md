# Held-out evaluation results

This folder separates the first blind run from later development work. The distinction matters because the first run is the only one that had not yet informed system changes.

## First live OpenRouter run (baseline)

The 20-question set was frozen before execution at SHA-256 `686dd4bed0fcdf585dc9cf028533d5f9a77d58bc3e58882457edf3bf14c2826f`. It contains 15 answerable questions and five cases where refusal is correct.

The historical manifest records `strategy: v2` and `generator: openrouter` because those were the names used when the blind run was frozen. These fields document that experiment; they are not selectable code paths in the final V4 package.

| Metric | Human-verified result |
|---|---:|
| Answer Correct | 14/15 (93.3%) |
| Source Supports Answer | 14/15 (93.3%) |
| Exact first-source match | 11/15 (73.3%) |
| Partial source credit | 12.5/15 (83.3%) |
| Correct Abstention | 5/5 (100%) |

The single answer failure was an over-conservative refusal. Three other answers used course pages that supported the claims but were not the single frozen gold page. Yujie reviewed every answer and source row and recorded reviewer ID `HYJ`.

Canonical files:

- `questions_holdout.csv`: frozen questions and gold fields;
- `results_holdout_v3.csv`: untouched model output;
- `results_holdout_v3.manifest.json`: model, prompt, corpus, database, input and output hashes;
- `scores_holdout_v3_verified.csv`: human-verified judgments and reasons;

## Optimized live OpenRouter rerun (final system)

Error analysis showed that long candidate pages were truncated from the beginning, sometimes removing the answer-bearing passage, and that a three-term lexical gate rejected one valid paraphrase before the model saw its evidence. V4 selects a query-focused window from long pages and lets two lexical anchors reach the evidence-constrained model. No question ID, expected answer, filename, or gold page is hard-coded.

| Metric | Human-verified result |
|---|---:|
| Answer Correct | 15/15 (100%) |
| Source Supports Answer | 15/15 (100%) |
| Exact first-source match | 13/15 (86.7%) |
| Partial source credit | 14/15 (93.3%) |
| Correct Abstention | 5/5 (100%) |

This is the final optimized system result. It is a live OpenRouter run, but not a second blind estimate because errors from the first run informed the change. The report therefore presents it as an optimized rerun rather than a new held-out test.

Canonical files:

- `results_holdout_v4_posthoc.csv` and its manifest;
- `scores_holdout_v4_posthoc_verified.csv`: human-verified post-hoc judgments.

## Observed manual Ctrl-F baseline

The project author manually attempted H01-H15, the same 15 answerable held-out questions, using ordinary PDF search and Ctrl-F. The frozen query was used once per question, with a three-minute stop rule. Elapsed time and the number of opened PDFs were recorded during the run.

| Metric | Observed result |
|---|---:|
| Answer Correct | 12/15 (80.0%) |
| Source Supports Answer | 12/15 (80.0%) |
| Exact frozen source | 8/15 (53.3%) |
| Partial source credit | 10/15 (66.7%) |
| Total time | 24:02 |
| Mean / median time | 1:36 / 1:38 |
| PDFs opened | 42 total; 2.8 per question |

H09 is deliberately scored as a timed failure. The initial search found some cost figures, but the exact slide and the `0.55`-to-`0.19` margin calculation were resolved only after the trial. H10 and H12 reached the three-minute limit without a supported answer.

The canonical row-level record is `ctrl_f_manual_results.csv`. This is an observed single-participant self-study, not the earlier automated 30-question proxy and not a population-level usability claim.

## Scoring definitions

- **Answer Correct:** strict binary correctness on answerable cases.
- **Source Supports Answer:** cited evidence entails the answer, including a valid alternative course page.
- **Exact first-source match:** the first citation equals the frozen filename and physical page.
- **Partial source credit:** 1 for an exact first-source match, 0.5 for a different course page that supports the answer, and 0 otherwise.
- **Correct Abstention:** refusal on an unanswerable case.
