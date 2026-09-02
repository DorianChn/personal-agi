"""将合并后的模型量化为 GGUF（Q4_K_M 等），供本地 Ollama 推理。

用法：
    python scripts/quantize.py --config config/config.yaml --model outputs/merged

说明：
    - 需要 llama.cpp（云端用 convert_hf_to_gguf.py + llama-quantize）
    - 也可用 Unsloth 的 save_pretrained_gguf 直接导出 GGUF
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model", required=True, help="合并后的模型目录")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    quant_type = cfg["quantization"]["quant_type"]
    out_dir = Path(cfg["quantization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 方案 A：Unsloth 直接导出 GGUF（推荐，无需手动跑 llama.cpp）
    try:
        from unsloth import FastLanguageModel

        print(f"=== 用 Unsloth 导出 GGUF（{quant_type}）===")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=cfg["training"]["max_seq_length"]
        )
        model.save_pretrained_gguf(
            str(out_dir),
            tokenizer,
            quantization_method=quant_type,
        )
        print(f"✅ GGUF 已导出至 {out_dir}")
        return
    except ImportError:
        pass

    # 方案 B：llama.cpp 工具链
    print("=== Unsloth 不可用，改用 llama.cpp ===")
    print("请手动执行：")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {args.model} --outtype f16")
    print(f"  llama-quantize <f16.gguf> {quant_type}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
