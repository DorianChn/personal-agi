import unittest
from unittest.mock import Mock, patch

from src.inference.client import InferenceClient


class FakeStreamingResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)


class OpenAIStreamingTests(unittest.TestCase):
    @patch("src.inference.client.requests.post")
    def test_streaming_openai_response_yields_content_deltas(self, post):
        post.return_value = FakeStreamingResponse(
            [
                b": keep-alive",
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        client = InferenceClient(
            backend="openai",
            base_url="https://example.test/v1",
            model="test-model",
            api_key="test-key",
        )

        result = client.chat([{"role": "user", "content": "Hi"}], stream=True)

        self.assertEqual(["Hello", " world"], list(result))
        post.assert_called_once()
        request = post.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertTrue(request["json"]["stream"])
        self.assertEqual("Bearer test-key", request["headers"]["Authorization"])

    @patch("src.inference.client.requests.post")
    def test_non_streaming_openai_response_remains_a_string(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        post.return_value = response
        client = InferenceClient(backend="openai")

        result = client.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual("Hello", result)
        self.assertNotIn("stream", post.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
