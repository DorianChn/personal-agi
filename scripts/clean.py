"""数据清洗脚本：把原始语料转换为标准 Alpaca 格式。

支持输入多种格式（jsonl / txt / csv），输出规范化后的训练数据。
用法：
    python scripts/clean.py --input data/raw --output data/processed --format jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize(rows: list[dict]) -> list[dict]:
    """标准化为 {instruction, input, output}，过滤无效样本。"""
    cleaned: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        instruction = str(r.get("instruction", "")).strip()
        inp = str(r.get("input", "")).strip()
        output = str(r.get("output", "")).strip()
        # 过滤空内容
        if not instruction or not output:
            continue
        # 去重（按 instruction+output）
        key = instruction + "\x00" + output
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"instruction": instruction, "input": inp, "output": output})
    return cleaned


def load_file(path: Path, fmt: str) -> list[dict]:
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if fmt == "csv":
        with path.open("r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if fmt == "txt":
        # txt 视为每行一条 instruction=output（用 | 分隔）
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                ins, out = line.split("|", 1)
                rows.append({"instruction": ins.strip(), "input": "", "output": out.strip()})
        return rows
    raise ValueError(f"不支持的格式：{fmt}")


def main() -> None:
    p = argparse.ArgumentParser(description="清洗并标准化训练数据")
    p.add_argument("--input", required=True, help="输入文件或目录")
    p.add_argument("--output", default="data/processed", help="输出目录")
    p.add_argument("--format", default="jsonl", choices=["jsonl", "csv", "txt"])
    args = p.parse_args()

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(src.glob(f"*.{args.format}"))

    all_rows: list[dict] = []
    for f in files:
        all_rows.extend(load_file(f, args.format))

    cleaned = normalize(all_rows)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 9:1 切分训练/验证
    split = int(len(cleaned) * 0.9)
    train, eval_set = cleaned[:split], cleaned[split:]

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(out_dir / "train.jsonl", train)
    write(out_dir / "eval.jsonl", eval_set)

    print(f"[ok] 清洗完成：原始 {len(all_rows)} 条 -> 有效 {len(cleaned)} 条")
    print(f"   训练集 {len(train)} 条，验证集 {len(eval_set)} 条")


if __name__ == "__main__":
    main()
