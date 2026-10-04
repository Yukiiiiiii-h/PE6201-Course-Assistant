"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import answer_question, format_answer
from .evaluate import merge_scores, run_evaluation, summarize_results
from .index import build_index, result_to_dict, search
from .openrouter import OpenRouterError


DEFAULT_DB = Path("data/processed/pe6201.sqlite")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pe6201-rag", description="PE6201 citation-first RAG — final V4 pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Extract PDFs and build the SQLite FTS index")
    build.add_argument("--materials", type=Path, default=Path("data/materials/CONTENT"))
    build.add_argument("--db", type=Path, default=DEFAULT_DB)
    build.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    find = commands.add_parser("search", help="Show retrieved evidence without generating an answer")
    find.add_argument("question")
    find.add_argument("--db", type=Path, default=DEFAULT_DB)
    find.add_argument("--top-k", type=int, default=5)
    ask = commands.add_parser("ask", help="Produce a grounded answer with validated page citations")
    ask.add_argument("question")
    ask.add_argument("--db", type=Path, default=DEFAULT_DB)
    ask.add_argument("--model", help="OpenRouter model slug; defaults to OPENROUTER_MODEL")
    evaluate = commands.add_parser("eval", help="Fill system outputs in a reviewer-scored evaluation CSV")
    evaluate.add_argument("--input", type=Path, default=Path("evals/heldout/questions_holdout.csv"))
    evaluate.add_argument("--output", type=Path, default=Path("evals/generated/results_holdout.csv"))
    evaluate.add_argument("--db", type=Path, default=DEFAULT_DB)
    evaluate.add_argument("--scores", type=Path, help="Optional human score CSV keyed by id")
    evaluate.add_argument("--model", help="OpenRouter model slug; defaults to OPENROUTER_MODEL")
    evaluate.add_argument("--resume", action="store_true", help="Continue an existing output CSV by id")
    score = commands.add_parser("merge-scores", help="Merge human scores without rerunning answers")
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--scores", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Summarize scored results and abstention fields")
    summarize.add_argument("--input", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build":
        manifest = build_index(args.materials, args.db, args.manifest)
        print(json.dumps(manifest["summary"], indent=2))
    elif args.command == "search":
        results = search(args.db, args.question, limit=args.top_k)
        print(json.dumps([result_to_dict(r) for r in results], indent=2, ensure_ascii=False))
    elif args.command == "ask":
        try:
            answer = answer_question(args.db, args.question, model=args.model)
        except OpenRouterError as exc:
            print(f"OpenRouter error: {exc}", file=sys.stderr)
            return 1
        print(format_answer(answer))
    elif args.command == "eval":
        try:
            run_evaluation(
                args.db, args.input, args.output, args.scores,
                args.model, args.resume,
            )
        except OpenRouterError as exc:
            print(f"OpenRouter error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {args.output}")
    elif args.command == "merge-scores":
        merge_scores(args.results, args.scores, args.output)
        print(f"Wrote {args.output}")
    elif args.command == "summarize":
        print(json.dumps(summarize_results(args.input), indent=2))
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
