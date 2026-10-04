# OpenRouter deployment notes

## Final V4 pipeline

The model does not search the whole course. Local retrieval first selects eight strong pages and expands around them to capture answer slides, diagrams, and explanations that sit nearby. The model sees at most 32 numbered evidence blocks.

The model must return strict JSON containing short claims and the evidence IDs for each claim. The application, not the model, owns the mapping from an evidence ID to the original filename and physical PDF page. If the model invents an ID or returns claims without evidence, the answer is rejected.

## Configuration

Required:

```bash
export OPENROUTER_API_KEY="..."
```

Optional:

```bash
export OPENROUTER_MODEL="google/gemini-3.8-flash"
export OPENROUTER_MAX_ESTIMATED_USD="2.00"
export OPENROUTER_BUDGET_LEDGER="data/processed/openrouter_budget.json"
export OPENROUTER_HTTP_REFERER="https://your-app.example"
```

The default endpoint is `https://openrouter.ai/api/v1/chat/completions`. `OPENROUTER_BASE_URL` may override it for testing, but production should use HTTPS.

The program keeps a persistent conservative reservation ledger and refuses an HTTP attempt before sending when the cumulative estimate would exceed `OPENROUTER_MAX_ESTIMATED_USD` (default: $2.00). Every retry is reserved separately because a malformed but successful provider response may still be billable. The default ledger is `data/processed/openrouter_budget.json` and is gitignored with other generated data.

This is an application guard, not an OpenRouter account-level spending limit. Provider billing remains authoritative. Reset the ledger only when deliberately beginning a new local budget period; deleting it does not alter or refund provider charges.

## Suggested release check

1. Test one known question and verify the answer, filename, and physical page.
2. Run an evaluation into a new CSV rather than overwriting submitted evidence.
3. Review every answer manually against `expected_answer_notes`; do not copy historical scores.
4. Record the exact model slug and run date.
5. Write a new held-out question set before making another blind accuracy claim, because the submitted sets have already influenced development.
6. Record the generated run manifest next to the CSV and have an independent reviewer fill the held-out score fields.

## Cost and data handling

The default model is configurable. At the time this integration was written, OpenRouter listed `google/gemini-3.8-flash` at $0.75 per million input tokens and $3.75 per million output tokens, with structured-output support. Prices and availability can change, so check the model page before a large run.

The final answer pipeline sends retrieved course passages to an external provider. API keys are read from environment variables and are never included in generated evaluation files. Local retrieval can be inspected with `pe6201-rag search`, but grounded answer generation requires OpenRouter.

Evidence blocks are explicitly delimited as untrusted data, and the model is instructed not to follow instructions contained in them. Clearly out-of-scope, live-data, private/medical, assignment-completion, and evidence-bypass requests are rejected before an API call.
