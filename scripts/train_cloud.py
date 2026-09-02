"""云端微调脚本（Unsloth QLoRA），在 AutoDL RTX 4090/A100 上训练 Qwen3.5-9B。

用法：
    python scripts/train_cloud.py --config config/config.yaml

说明：
    - 需先安装依赖：pip install -r docker/requirements-train.txt
    - 9B 模型用 QLoRA 4bit 训练，24GB 显存可轻松承载。
    - 训练产物（LoRA adapter）输出到 outputs/<run_name>。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="云端 QLoRA 微调 9B 模型")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--base_model", default=None, help="覆盖配置中的基座 HF id")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    base_hf = args.base_model or cfg["base_model"]["hf_id"]

    # 训练库导入放在此处，避免本地无 GPU 环境时 import 报错
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template

    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(t["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动适配硬件精度：V100 等 Volta 卡不支持 bf16，回退 fp16
    import torch
    use_bf16 = is_bfloat16_supported()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "V100" in gpu_name or "T4" in gpu_name or "P100" in gpu_name:
            use_bf16 = False
    print(f"=== 设备：{torch.cuda.get_device_name(0)}，使用 {'bf16' if use_bf16 else 'fp16'} ===")

    print(f"=== 加载基座模型：{base_hf} ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_hf,
        max_seq_length=t["max_seq_length"],
        dtype=None,                      # 自动选择 bf16/fp16
        load_in_4bit=t["use_4bit"],
    )

    print("=== 应用 LoRA ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=t["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    def formatting_prompts_func(examples):
        """将 Alpaca 格式转为 ChatML 对话格式。"""
        texts = []
        for ins, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            user_content = ins + (f"\n{inp}" if inp else "")
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))
        return {"text": texts}

    print("=== 加载数据 ===")
    dataset = load_dataset(
        "json",
        data_files={
            "train": t["train_file"],
            "eval": t["eval_file"],
        },
    )
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            num_train_epochs=t["num_epochs"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler"],
            warmup_ratio=t["warmup_ratio"],
            logging_steps=t["logging_steps"],
            save_steps=t["save_steps"],
            eval_strategy=t["eval_strategy"],
            eval_steps=t["eval_steps"],
            save_total_limit=t["save_total_limit"],
            bf16=use_bf16,
            fp16=not use_bf16,
            optim="adamw_8bit",
            seed=42,
            max_seq_length=t["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
        ),
    )

    print("=== 开始训练 ===")
    trainer.train()

    # 保存 LoRA adapter
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"=== 训练完成，LoRA adapter 已保存至 {adapter_dir} ===")


if __name__ == "__main__":
    main()
