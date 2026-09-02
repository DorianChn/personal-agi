"""本地微调脚本（transformers + peft QLoRA），针对 RTX 5060 8GB 显存调参。

用法：
    python scripts/train_local.py --model Qwen/Qwen3-0.6B --epochs 3
    python scripts/train_local.py --model Qwen/Qwen3-1.5B --epochs 2

说明：
    - 使用 4bit QLoRA + 梯度检查点，8GB 显存可训 0.6B（轻松）与 1.5B（较紧）。
    - 与 train_cloud.py（Unsloth，云端）互补，本脚本用于本地验证全流程。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# 国内网络优先使用 HF 镜像（避免 huggingface.co 下载中断）
if os.environ.get("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 参数映射：不同模型的显存调参预设
PRESETS = {
    "0.6b": {
        "max_seq_length": 512,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "lora_r": 8,
        "lora_alpha": 16,
    },
    "1.7b": {
        "max_seq_length": 384,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_r": 8,
        "lora_alpha": 16,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 QLoRA 微调")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF 基座模型 id")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_file", default="data/processed/train.jsonl")
    p.add_argument("--eval_file", default="data/processed/eval.jsonl")
    p.add_argument("--output_dir", default="outputs/local")
    p.add_argument("--preset", default=None, choices=list(PRESETS), help="显存调参预设")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    assert torch.cuda.is_available(), "未检测到 CUDA GPU，请确认 torch 为 CUDA 版"
    print(f"=== GPU: {torch.cuda.get_device_name(0)}，显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB ===")

    # 自动选 preset（默认按模型名）
    preset_name = args.preset or ("1.7b" if ("1.5" in args.model or "1.7" in args.model) else "0.6b")
    preset = PRESETS[preset_name]
    print(f"=== 调参预设：{preset_name} ===")

    # run_name：本地路径取末两段，HF 名取模型名
    parts = [p for p in args.model.replace("\\", "/").split("/") if p]
    if "models" in parts or "Qwen" in args.model:
        tag = parts[-2] if len(parts) >= 2 and parts[-1] in ("master", "snapshots") else parts[-1]
    else:
        tag = parts[-1]
    run_name = f"{tag}-{datetime.now().strftime('%m%d-%H%M')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 显存策略：0.6B/1.5B 用 fp16 全精度 + LoRA，8GB 显存足够且最稳定
    # （避免 bitsandbytes 4bit 在 Blackwell 架构上的混合精度兼容问题）
    print(f"=== 加载基座：{args.model} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=preset["lora_r"],
        lora_alpha=preset["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 数据格式化：Alpaca → 文本
    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)

    def fmt(rows):
        """每条样本返回 {"text": ...}，Dataset.from_list 需要 list of dict。"""
        out = []
        for r in rows:
            ins = r["instruction"] + (f"\n{r['input']}" if r.get("input") else "")
            out.append({"text": f"### 指令:\n{ins}\n\n### 回答:\n{r['output']}"})
        return out

    train_ds = Dataset.from_list(fmt(train_rows))
    eval_ds = Dataset.from_list(fmt(eval_rows))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=preset["per_device_train_batch_size"],
        gradient_accumulation_steps=preset["gradient_accumulation_steps"],
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        bf16=False,
        logging_steps=5,
        save_strategy="epoch",
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        max_length=preset["max_seq_length"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args,
    )

    print("=== 开始训练 ===")
    trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"=== 训练完成，LoRA adapter 保存至 {adapter_dir} ===")


if __name__ == "__main__":
    main()
