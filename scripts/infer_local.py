"""加载本地微调模型（LoRA）进行推理验证。

用法：
    python scripts/infer_local.py --base D:/models/models/Qwen--Qwen3-0.6B/snapshots/master --adapter outputs/local/qwen3-0.6b-finetuned/lora_adapter
    python scripts/infer_local.py --base D:/models/models/Qwen--Qwen3-1.7B/snapshots/master --adapter outputs/local/qwen3-1.7b-finetuned/lora_adapter
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="本地微调模型推理")
    p.add_argument("--base", required=True, help="基座模型路径")
    p.add_argument("--adapter", required=True, help="LoRA adapter 路径")
    p.add_argument("--prompt", default="用一句话介绍你自己")
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n=== 模型回复 ===\n{reply}\n")


if __name__ == "__main__":
    main()
