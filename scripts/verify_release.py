"""Run offline release checks without an API key or corpus rebuild."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = str(ROOT / "src")


def run(*arguments: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = PYTHONPATH
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def check_csv(path: Path, expected_rows: int) -> None:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows:
        raise AssertionError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{path} contains duplicate IDs")


def main() -> int:
    required = [
        ROOT / "README.md",
        ROOT / "data/materials/CONTENT",
        ROOT / "data/processed/manifest.json",
        ROOT / "data/processed/pe6201.sqlite",
        ROOT / "evals/heldout/questions_holdout.csv",
        ROOT / "evals/heldout/ctrl_f_manual_results.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing required release files: {missing}")

    manifest = json.loads((ROOT / "data/processed/manifest.json").read_text(encoding="utf-8"))
    expected_summary = {
        "pdfs": 85,
        "pages": 1558,
        "extracted_pages": 1553,
        "empty_pages": 5,
        "chunks": 2260,
        "errors": 0,
    }
    if manifest.get("summary") != expected_summary:
        raise AssertionError(f"Unexpected corpus summary: {manifest.get('summary')}")

    with sqlite3.connect(ROOT / "data/processed/pe6201.sqlite") as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if (document_count, chunk_count) != (85, 2260):
        raise AssertionError(
            f"Unexpected index size: {document_count} documents, {chunk_count} chunks"
        )

    check_csv(ROOT / "evals/heldout/questions_holdout.csv", 20)
    check_csv(ROOT / "evals/heldout/ctrl_f_manual_results.csv", 15)
    check_csv(ROOT / "evals/heldout/results_holdout_v3.csv", 20)
    check_csv(ROOT / "evals/heldout/results_holdout_v4_posthoc.csv", 20)

    run("-m", "unittest", "discover", "-s", "tests", "-v")
    print("Unit tests passed.")
    retrieved = json.loads(run(
        "-m", "pe6201_rag", "search",
        "What are the three stages of the RAG workflow?",
        "--top-k", "12",
    ))
    if not any(
        result["source_file"] == "PE6201_Class2_C3_RAG.pdf"
        and result["page"] == 3
        and "embed" in result["text"].lower()
        for result in retrieved
    ):
        raise AssertionError("Included-corpus retrieval smoke test failed")
    refusal = run(
        "-m", "pe6201_rag", "ask",
        "What is today's weather?",
    )
    if "Grounded: No" not in refusal or "Sources:\n- None" not in refusal:
        raise AssertionError("Included-corpus abstention smoke test failed")

    root_help = run("-m", "pe6201_rag", "--help")
    ask_help = run("-m", "pe6201_rag", "ask", "--help")
    eval_help = run("-m", "pe6201_rag", "eval", "--help")
    if "baseline" in root_help:
        raise AssertionError("Automated Ctrl-F proxy remains in the final CLI")
    if any(flag in ask_help + eval_help for flag in ("--strategy", "--generator")):
        raise AssertionError("Old version-selection flags remain in the final CLI")

    print("Release verification passed: files, corpus, index, CSVs, tests, V4 retrieval, and abstention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
