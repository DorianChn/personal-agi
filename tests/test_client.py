from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from src.inference.client import InferenceClient


class InferenceClientTests(unittest.TestCase):
    def test_openai_stream_yields_sse_content_chunks(self) -> None:
        response = Mock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client = InferenceClient(backend="openai", api_key="test")

        with patch("src.inference.client.requests.post", return_value=response) as post:
            chunks = list(client.chat([{"role": "user", "content": "hi"}], stream=True))

        self.assertEqual(chunks, ["hello", " world"])
        post.assert_called_once()
        self.assertTrue(post.call_args.kwargs["stream"])

    def test_openai_stream_accepts_jsonl_and_reasoning_content(self) -> None:
        response = Mock()
        response.iter_lines.return_value = [
            json.dumps({"choices": [{"delta": {"reasoning_content": "think"}}]}),
            json.dumps({"choices": [{"delta": {"content": "answer"}}]}),
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client = InferenceClient(backend="openai", api_key="test")

        with patch("src.inference.client.requests.post", return_value=response):
            self.assertEqual(
                list(client.chat([{"role": "user", "content": "hi"}], stream=True)),
                ["think", "answer"],
            )


if __name__ == "__main__":
    unittest.main()
