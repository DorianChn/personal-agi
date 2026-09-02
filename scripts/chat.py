"""命令行对话入口。"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.client import get_client  # noqa: E402


def main() -> None:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen = cfg.get("generation", {})
    client = get_client(cfg)

    print(f"🤖 已连接模型：{client.model}（输入 quit 退出）\n")
    messages: list[dict[str, str]] = []
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break
        messages.append({"role": "user", "content": user})
        print("\n助手：", end="", flush=True)
        resp = client.chat(
            messages,
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.9),
            max_tokens=gen.get("max_tokens", 1024),
        )
        print(resp + "\n")
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
