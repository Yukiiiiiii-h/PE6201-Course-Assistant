"""Evaluation file helpers for independent reviewer scoring."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .answer import OPENROUTER_PROMPT_VERSION, answer_question
from .openrouter import DEFAULT_MODEL


FIELDS = [
    "id", "status", "question", "expected_answer_notes", "expected_source_file", "expected_page",
    "system_answer", "system_sources", "answer_correct_0_or_1", "source_correct_0_or_1",
    "ctrl_f_query", "ctrl_f_answer", "ctrl_f_source_file", "ctrl_f_page",
    "ctrl_f_answer_correct_0_or_1", "ctrl_f_source_correct_0_or_1", "notes",
    "answerable_0_or_1", "expected_behavior", "author_provenance", "review_status",
    "source_supports_answer_0_or_1", "correct_abstention_0_or_1", "reviewer_id",
    "grounded", "confidence", "external_knowledge", "generator", "model",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_run_manifest(
    *, db_path: Path, input_csv: Path, output_csv: Path, question_count: int,
    model: str | None,
) -> None:
    corpus_manifest = db_path.with_name("manifest.json")
    selected_model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "input_csv": str(input_csv),
        "input_sha256": _sha256(input_csv),
        "output_csv": str(output_csv),
        "output_sha256": _sha256(output_csv),
        "database": str(db_path),
        "database_sha256": _sha256(db_path),
        "corpus_manifest": str(corpus_manifest) if corpus_manifest.exists() else None,
        "corpus_manifest_sha256": _sha256(corpus_manifest) if corpus_manifest.exists() else None,
        "question_count": question_count,
        "pipeline": "v4-final",
        "generator": "openrouter",
        "model": selected_model,
        "prompt_version": OPENROUTER_PROMPT_VERSION,
        "temperature": 0,
        "pricing_note": "Estimated local guard only; provider billing is authoritative.",
    }
    path = output_csv.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_results(results_csv: Path) -> dict[str, int | float | None]:
    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    def binary_total(field: str) -> tuple[int, int]:
        values = [row.get(field, "") for row in rows]
        scored = [int(value) for value in values if value in {"0", "1"}]
        return sum(scored), len(scored)

    answer_correct, answer_n = binary_total("answer_correct_0_or_1")
    source_correct, source_n = binary_total("source_correct_0_or_1")
    source_support, support_n = binary_total("source_supports_answer_0_or_1")
    correct_abstentions, abstention_n = binary_total("correct_abstention_0_or_1")
    grounded_values = [row.get("grounded", "").strip().lower() for row in rows]
    observed = [value for value in grounded_values if value in {"true", "false"}]
    abstentions = sum(value == "false" for value in observed)
    selective_scores = [
        int(row["answer_correct_0_or_1"])
        for row in rows
        if row.get("grounded", "").strip().lower() == "true"
        and row.get("answer_correct_0_or_1", "") in {"0", "1"}
    ]
    return {
        "rows": len(rows),
        "answer_correct": answer_correct,
        "answer_scored": answer_n,
        "source_correct": source_correct,
        "source_scored": source_n,
        "source_supports_answer": source_support,
        "source_support_scored": support_n,
        "abstentions": abstentions,
        "grounding_observed": len(observed),
        "abstention_rate": abstentions / len(observed) if observed else None,
        "correct_abstentions": correct_abstentions,
        "correct_abstention_scored": abstention_n,
        "correct_abstention_rate": (
            correct_abstentions / abstention_n if abstention_n else None
        ),
        "selective_accuracy": (
            sum(selective_scores) / len(selective_scores) if selective_scores else None
        ),
        "selective_accuracy_scored": len(selective_scores),
    }


def merge_scores(results_csv: Path, scores_csv: Path, output_csv: Path) -> None:
    """Merge reviewer scores into existing outputs without rerunning a generator."""
    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with scores_csv.open(newline="", encoding="utf-8-sig") as stream:
        scores = {row["id"]: row for row in csv.DictReader(stream)}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            merged = {field: row.get(field, "") for field in FIELDS}
            score = scores.get(merged["id"], {})
            merged["answer_correct_0_or_1"] = score.get("answer_correct_0_or_1", "")
            merged["source_correct_0_or_1"] = score.get("source_correct_0_or_1", "")
            merged["notes"] = score.get("reviewer_notes", score.get("notes", merged["notes"]))
            writer.writerow(merged)


def run_evaluation(
    db_path: Path,
    input_csv: Path,
    output_csv: Path,
    scores_csv: Path | None = None,
    model: str | None = None,
    resume: bool = False,
) -> None:
    with input_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if any((value or "").strip() for value in row.values())]
    scores: dict[str, dict[str, str]] = {}
    if scores_csv and scores_csv.exists():
        with scores_csv.open(newline="", encoding="utf-8-sig") as stream:
            scores = {row["id"]: row for row in csv.DictReader(stream)}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if resume and output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8-sig") as stream:
            completed_ids = {row["id"] for row in csv.DictReader(stream)}
    mode = "a" if completed_ids else "w"
    with output_csv.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(rows, start=1):
            merged = {field: row.get(field, "") for field in FIELDS}
            merged["id"] = row.get("id") or str(index)
            if merged["id"] in completed_ids:
                continue
            if row.get("question", "").strip():
                answer = answer_question(db_path, row["question"], model=model)
                merged["system_answer"] = answer.answer
                merged["system_sources"] = "; ".join(f"{c.source_file} p.{c.page}" for c in answer.citations)
                merged["grounded"] = "true" if answer.grounded else "false"
                merged["confidence"] = answer.confidence
                merged["external_knowledge"] = answer.external_knowledge
                merged["generator"] = "openrouter"
                merged["model"] = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
                if merged.get("answerable_0_or_1") == "0":
                    merged["correct_abstention_0_or_1"] = "1" if not answer.grounded else "0"
            if merged["id"] in scores:
                score = scores[merged["id"]]
                merged["answer_correct_0_or_1"] = score.get("answer_correct_0_or_1", "")
                merged["source_correct_0_or_1"] = score.get("source_correct_0_or_1", "")
                merged["notes"] = score.get("notes", merged["notes"])
            writer.writerow(merged)
            stream.flush()
    _write_run_manifest(
        db_path=db_path,
        input_csv=input_csv,
        output_csv=output_csv,
        question_count=len(rows),
        model=model,
    )
