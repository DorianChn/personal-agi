"""Regression tests for the HTTP API boundary."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeClient:
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], float, int]] = []

    def chat(self, messages, *, temperature, max_tokens):
        self.calls.append((messages, temperature, max_tokens))
        return "test response"


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Import outside the repository root to catch cwd-dependent config loads.
        cls._old_cwd = Path.cwd()
        os.chdir(Path(__file__).resolve().parent)
        from app import api

        cls.api = api

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._old_cwd)

    def test_health_reports_configured_model(self) -> None:
        with patch.object(self.api, "client", _FakeClient()):
            self.assertEqual(self.api.health(), {"status": "ok", "model": "test-model"})

    def test_explicit_zero_generation_values_are_preserved(self) -> None:
        fake = _FakeClient()
        request = self.api.ChatRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=0,
        )

        with patch.object(self.api, "client", fake):
            response = self.api.chat_completions(request)

        self.assertEqual(response["choices"][0]["message"]["content"], "test response")
        self.assertEqual(fake.calls[0][1:], (0, 0))


if __name__ == "__main__":
    unittest.main()
