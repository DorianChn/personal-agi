"""将 GGUF 模型导入本地 Ollama。

用法：
    python scripts/import_ollama.py --gguf outputs/gguf/<file>.gguf --name personal-agi-9b
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="GGUF 文件路径")
    p.add_argument("--name", default="personal-agi-9b", help="Ollama 模型名")
    p.add_argument("--system", default="你是我的个人 AI 助手。", help="system prompt")
    args = p.parse_args()

    gguf = Path(args.gguf)
    if not gguf.exists():
        raise FileNotFoundError(f"GGUF 文件不存在：{gguf}")

    modelfile = (
        f"FROM {gguf.resolve()}\n"
        f'SYSTEM """{args.system}"""\n'
        f'TEMPLATE """{{{{ if .System }}}}<|im_start|>system\n{{{{ .System }}}}<|im_end|>\n{{{{ end }}}}<|im_start|>user\n{{{{ .Prompt }}}}<|im_end|>\n<|im_start|>assistant\n"""\n'
    )
    mf_path = Path("Modelfile.personal-agi")
    mf_path.write_text(modelfile, encoding="utf-8")

    print(f"=== 导入 Ollama：{args.name} ===")
    subprocess.run(["ollama", "create", args.name, "-f", str(mf_path)], check=True)
    print(f"✅ 已导入，可用 `ollama run {args.name}` 对话")


if __name__ == "__main__":
    main()
