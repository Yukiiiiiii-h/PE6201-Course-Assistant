# Evaluation files

This folder contains only the final held-out evaluation evidence needed to understand and reproduce the submitted results. Prototype questions, the 30-question development regression, earlier V1/V2 runs, automated Ctrl-F proxy outputs, tuning comparisons and duplicate merged files are intentionally excluded from the instructor ZIP.

## What is included

### Held-out final comparison

- `heldout/questions_holdout.csv` and `.sha256`: the frozen 20-question set, containing 15 answerable and five refusal cases.
- `heldout/freeze_record.md`: when and how the set was frozen.
- `heldout/results_holdout_v3.csv` and its manifest: untouched first blind OpenRouter output.
- `heldout/scores_holdout_v3_verified.csv`: Yujie's human review of the blind run.
- `heldout/results_holdout_v4_posthoc.csv` and its manifest: optimized live rerun after error analysis.
- `heldout/scores_holdout_v4_posthoc_verified.csv`: Yujie's human review of the optimized rerun.
- `heldout/ctrl_f_manual_results.csv`: observed 15-question manual Ctrl-F study.
- `heldout/RESULTS.md`: metric definitions, comparison and limitations.

The `v3` result is retained as the untouched first blind run. The runnable program contains only final V4 code; no V3 implementation is included.

## Scoring

- **Answer Correct:** binary correctness on answerable cases.
- **Source Supports Answer:** whether a cited PE6201 page genuinely supports the answer, including a valid alternative page.
- **Exact Source:** whether the first citation exactly matches the frozen filename and physical page.
- **Partial Source:** 1.0 for the exact page, 0.5 for a different supporting course page, and 0 otherwise.
- **Correct Abstention:** refusal on an unanswerable case.

The first held-out run is the blind estimate. The V4 rerun is labelled post-hoc because the first-run errors informed the improvements. The manual Ctrl-F result is a one-participant self-study and is not presented as population-level user research.

## Run a new final-V4 evaluation

Install the package and configure OpenRouter as described in the root `README.md`, then write the new output under `evals/generated/`:

```bash
pe6201-rag eval \
  --model "$OPENROUTER_MODEL" \
  --input evals/heldout/questions_holdout.csv \
  --output evals/generated/results_holdout_new.csv
```

Every run writes an adjacent manifest with the model, prompt version and hashes of the input, output, database and corpus manifest. Score fields remain blank until a human checks the answer and source; the system does not grade itself.

To merge a completed reviewer sheet without rerunning the model:

```bash
pe6201-rag merge-scores \
  --results evals/generated/results_holdout_new.csv \
  --scores path/to/reviewer_scores.csv \
  --output evals/generated/results_holdout_new_scored.csv
```

Run offline checks with:

```bash
python -m unittest discover -s tests -v
python scripts/verify_release.py
```
