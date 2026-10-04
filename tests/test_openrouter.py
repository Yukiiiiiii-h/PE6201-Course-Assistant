import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from pe6201_rag.openrouter import OpenRouterError, chat_json


class _FakeResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.temp.name) / "budget.json"
        self.environment = patch.dict(
            os.environ,
            {"OPENROUTER_BUDGET_LEDGER": str(self.ledger)},
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_missing_key_fails_without_network(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(OpenRouterError):
                chat_json([], {"type": "object"})

    def test_local_budget_can_block_before_network(self):
        with patch.dict("os.environ", {"OPENROUTER_MAX_ESTIMATED_USD": "0"}, clear=True):
            with self.assertRaisesRegex(OpenRouterError, "budget"):
                chat_json(
                    [{"role": "user", "content": "test"}],
                    {"type": "object"},
                    api_key="not-a-real-key",
                )

    @patch("urllib.request.urlopen")
    def test_structured_response_is_parsed(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": '{"grounded": true}'}}]}
        )
        result = chat_json(
            [{"role": "user", "content": "test"}],
            {"type": "object"},
            api_key="not-a-real-key",
        )
        self.assertEqual(result, {"grounded": True})

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_invalid_json_is_retried(self, mock_urlopen, _mock_sleep):
        mock_urlopen.side_effect = [
            _FakeResponse({"choices": [{"message": {"content": "not-json"}}]}),
            _FakeResponse({"choices": [{"message": {"content": '{"grounded": true}'}}]}),
        ]
        result = chat_json(
            [{"role": "user", "content": "test"}],
            {"type": "object"},
            api_key="not-a-real-key",
        )
        self.assertEqual(result, {"grounded": True})
        self.assertEqual(mock_urlopen.call_count, 2)
        ledger = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertGreater(ledger["reserved_estimated_usd"], 0.02)

    @patch("urllib.request.urlopen")
    def test_budget_is_persistent_across_calls(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": '{"grounded": true}'}}]}
        )
        with patch.dict(os.environ, {"OPENROUTER_MAX_ESTIMATED_USD": "0.02"}):
            chat_json(
                [{"role": "user", "content": "test"}],
                {"type": "object"},
                api_key="not-a-real-key",
            )
            with self.assertRaisesRegex(OpenRouterError, "budget"):
                chat_json(
                    [{"role": "user", "content": "test"}],
                    {"type": "object"},
                    api_key="not-a-real-key",
                )
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_http_429_is_retried(self, mock_urlopen, _mock_sleep):
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("https://example.invalid", 429, "rate limit", {}, None),
            _FakeResponse({"choices": [{"message": {"content": '{"grounded": true}'}}]}),
        ]
        result = chat_json(
            [{"role": "user", "content": "test"}],
            {"type": "object"},
            api_key="not-a-real-key",
        )
        self.assertEqual(result, {"grounded": True})
        self.assertEqual(mock_urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
