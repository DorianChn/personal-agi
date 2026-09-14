"""合并 LoRA adapter 到基座模型，输出完整权重。

用法：
    python scripts/merge.py --config config/config.yaml --adapter outputs/<run>/lora_adapter
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--adapter", required=True, help="LoRA adapter 目录")
    p.add_argument("--output", default="outputs/merged", help="合并后输出目录")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_hf = cfg["base_model"]["hf_id"]

    from unsloth import FastLanguageModel

    print(f"=== 加载基座 {base_hf} 并合并 LoRA ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_hf,
        max_seq_length=cfg["training"]["max_seq_length"],
        load_in_4bit=False,
    )
    model = model.load_adapter(args.adapter)

    # 合并并保存为 fp16 完整权重
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = model.merge_and_unload()
    model.save_pretrained(str(out), save_method="merged_16bit")
    tokenizer.save_pretrained(str(out))
    print(f"[ok] 合并完成，完整权重保存至 {out}")


if __name__ == "__main__":
    main()
