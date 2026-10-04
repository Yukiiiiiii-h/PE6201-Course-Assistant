"""Small, dependency-free OpenRouter client with strict JSON responses."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi


DEFAULT_MODEL = "google/gemini-3.8-flash"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BUDGET_USD = 2.0
_MODEL_PRICING_PER_MILLION = {
    # Snapshot used only for a conservative local guard. Provider billing is
    # authoritative and can change; use a server-side key limit as well when available.
    "google/gemini-3.8-flash": (0.75, 3.75),
}
DEFAULT_BUDGET_LEDGER = Path("data/processed/openrouter_budget.json")


class OpenRouterError(RuntimeError):
    """Raised for safe-to-display OpenRouter integration failures."""


def _reserve_estimated_budget(model: str, messages: list[dict[str, str]], max_tokens: int) -> float:
    """Persist a conservative cost reservation before every HTTP attempt."""
    budget = float(os.environ.get("OPENROUTER_MAX_ESTIMATED_USD", str(DEFAULT_BUDGET_USD)))
    if model in _MODEL_PRICING_PER_MILLION:
        input_rate, output_rate = _MODEL_PRICING_PER_MILLION[model]
    else:
        try:
            input_rate = float(os.environ["OPENROUTER_INPUT_USD_PER_M"])
            output_rate = float(os.environ["OPENROUTER_OUTPUT_USD_PER_M"])
        except (KeyError, ValueError) as exc:
            raise OpenRouterError(
                "No local pricing is configured for this model. Set "
                "OPENROUTER_INPUT_USD_PER_M and OPENROUTER_OUTPUT_USD_PER_M."
            ) from exc
    # Three characters per token deliberately overestimates typical English
    # course text. Output reserves the full configured maximum.
    input_chars = len(json.dumps(messages, ensure_ascii=False))
    estimated_input_tokens = input_chars / 3.0
    estimate = estimated_input_tokens * input_rate / 1_000_000 + max_tokens * output_rate / 1_000_000
    ledger_path = Path(os.environ.get("OPENROUTER_BUDGET_LEDGER", str(DEFAULT_BUDGET_LEDGER)))
    reserved = 0.0
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            reserved = float(ledger["reserved_estimated_usd"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OpenRouterError(
                f"Budget ledger {ledger_path} is unreadable; request not sent."
            ) from exc
    if reserved + estimate > budget:
        raise OpenRouterError(
            f"Local ${budget:.2f} estimated budget would be exceeded; request not sent."
        )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    updated = {
        "schema_version": 1,
        "reserved_estimated_usd": reserved + estimate,
        "budget_usd": budget,
        "last_model": model,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Conservative local reservations, not authoritative provider charges.",
    }
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, ledger_path)
    except OSError as exc:
        raise OpenRouterError("Could not persist the local budget reservation; request not sent.") from exc
    return estimate


def chat_json(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
    retries: int = 2,
) -> dict[str, Any]:
    """Call OpenRouter and parse one strict JSON object.

    Secrets are accepted only as an argument or environment variable and are
    never included in exception messages.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Add it to the process environment; "
            "do not commit it to the repository."
        )
    selected_model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    endpoint = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_ENDPOINT)
    # Reasoning-capable providers may count internal reasoning against this
    # ceiling before emitting the comparatively small JSON object.
    max_tokens = 3000
    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pe6201_grounded_answer",
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {"require_parameters": True},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "PE6201 Course Assistant",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "")
    if referer:
        headers["HTTP-Referer"] = referer
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    tls_context = ssl.create_default_context(cafile=certifi.where())

    for attempt in range(retries + 1):
        # Reserve every HTTP attempt. A provider may bill a successful response
        # even when its content is malformed and the application retries it.
        _reserve_estimated_budget(selected_model, messages, max_tokens)
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=tls_context) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            content = response_data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise OpenRouterError("OpenRouter returned a non-text structured response.")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise OpenRouterError("OpenRouter returned JSON that was not an object.")
            return parsed
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise OpenRouterError(f"OpenRouter request failed with HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            reason = getattr(exc, "reason", None)
            detail = type(reason).__name__ if reason is not None else type(exc).__name__
            raise OpenRouterError(f"Could not reach OpenRouter ({detail}).") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise OpenRouterError("OpenRouter returned an invalid structured response.") from exc
    raise OpenRouterError("OpenRouter request failed after retries.")
