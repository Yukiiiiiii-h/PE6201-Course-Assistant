import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pe6201_rag.answer import Answer, Citation
from pe6201_rag.evaluate import FIELDS, run_evaluation, summarize_results
from pe6201_rag.index import SCHEMA


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "test.sqlite"
        connection = sqlite3.connect(self.db)
        connection.executescript(SCHEMA)
        cursor = connection.execute(
            "INSERT INTO documents(source_path, source_file, sha256, page_count, extracted_pages) VALUES (?, ?, ?, ?, ?)",
            ("slides/test.pdf", "test.pdf", "abc", 1, 1),
        )
        connection.execute(
            "INSERT INTO chunks(document_id, page, chunk_index, text) VALUES (?, ?, ?, ?)",
            (
                cursor.lastrowid,
                1,
                0,
                "Retrieve relevant text at query time and put it in the prompt. Retrieval supplies facts and current context.",
            ),
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.temp.cleanup()

    @patch("pe6201_rag.evaluate.answer_question")
    def test_evaluation_records_grounding_and_manifest(self, mock_answer):
        input_csv = self.root / "questions.csv"
        output_csv = self.root / "results.csv"
        with input_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "id": "H01",
                    "status": "frozen",
                    "question": "How does retrieval supply facts and current context?",
                }
            )
        mock_answer.return_value = Answer(
            question="How does retrieval supply facts and current context?",
            answer="Retrieval supplies facts and current context. [S1]",
            citations=(Citation("S1", "test.pdf", "slides/test.pdf", 1, "Retrieval supplies facts."),),
            grounded=True,
            external_knowledge="None used.",
            confidence="model_grounded",
        )
        run_evaluation(self.db, input_csv, output_csv, model="test/model")
        with output_csv.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["grounded"], "true")
        self.assertEqual(row["external_knowledge"], "None used.")
        self.assertEqual(row["model"], "test/model")
        manifest = json.loads(
            output_csv.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["question_count"], 1)
        self.assertEqual(manifest["generator"], "openrouter")
        self.assertEqual(manifest["pipeline"], "v4-final")
        self.assertTrue(manifest["database_sha256"])

    def test_summary_reports_abstention_fields(self):
        results = self.root / "scored.csv"
        with results.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "id": "A",
                    "grounded": "false",
                    "correct_abstention_0_or_1": "1",
                    "answer_correct_0_or_1": "1",
                }
            )
            writer.writerow(
                {"id": "B", "grounded": "true", "answer_correct_0_or_1": "1"}
            )
        summary = summarize_results(results)
        self.assertEqual(summary["abstentions"], 1)
        self.assertEqual(summary["abstention_rate"], 0.5)
        self.assertEqual(summary["correct_abstentions"], 1)


if __name__ == "__main__":
    unittest.main()
