"""推理客户端：封装 Ollama 与 OpenAI 兼容接口的统一调用。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class InferenceClient:
    """统一推理客户端，支持 ollama 与 openai 两种后端。"""

    backend: str = "ollama"
    host: str = "http://localhost:11434"
    model: str = "personal-agi-9b"
    api_key: str = "EMPTY"
    base_url: str = "http://localhost:8000/v1"

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str | Any:
        if self.backend == "ollama":
            return self._chat_ollama(messages, temperature, top_p, max_tokens, stream)
        return self._chat_openai(messages, temperature, top_p, max_tokens, stream)

    def _chat_ollama(self, messages, temperature, top_p, max_tokens, stream):
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if stream:
            return self._stream_ollama(url, payload)
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        msg = resp.json()["message"]
        # 兼容思考型模型：优先返回 content，为空时回退到 thinking
        return msg.get("content") or msg.get("thinking") or ""

    def _stream_ollama(self, url, payload):
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                msg = data.get("message", {})
                # 流式下 content 与 thinking 分别输出，优先 content
                chunk = msg.get("content") or msg.get("thinking") or ""
                if chunk:
                    yield chunk

    def _chat_openai(self, messages, temperature, top_p, max_tokens, stream):
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_client(cfg: dict) -> InferenceClient:
    """根据 config 构造客户端。"""
    inf = cfg.get("inference", {})
    gen = cfg.get("generation", {})
    if inf.get("backend") == "openai":
        o = inf["openai"]
        return InferenceClient(
            backend="openai",
            base_url=o["base_url"],
            model=o["model"],
            api_key=o.get("api_key", "EMPTY"),
        )
    o = inf.get("ollama", {})
    return InferenceClient(
        backend="ollama",
        host=o.get("host", "http://localhost:11434"),
        model=o.get("model", "personal-agi-9b"),
    )
