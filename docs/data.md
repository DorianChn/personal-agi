# 数据收集与预处理

## 一、数据格式

统一使用 **Alpaca 格式**（JSONL，每行一个 JSON）：

```json
{"instruction": "用一句话解释什么是大语言模型。", "input": "", "output": "大语言模型是一种……"}
```

- `instruction`：任务指令（必填）
- `input`：补充输入/上下文（可空）
- `output`：期望回答（必填）

## 二、数据来源

按实用度排序：

1. **你的真实工作产出**（最有价值）
   - 写作：你写的小说章节、文案、复盘
   - 量化：你的策略笔记、行情研判
   - 对话：你与 AI 的高质量问答（脱敏后）

2. **公开指令数据**（冷启动）
   - Alpaca-zh、Firefly、BELLE 等中文指令数据集

3. **回流数据**：模型上线后收集真实使用对话，持续迭代。

## 三、预处理流程

```
data/raw/（原始语料）
   │  scripts/clean.py
   ▼
标准化 {instruction, input, output}
   │  去空、去重、格式校验
   ▼
data/processed/train.jsonl + eval.jsonl（9:1）
```

清洗规则（`scripts/clean.py` 已实现）：
- 过滤空 instruction / output
- 按 instruction+output 去重
- 统一字段结构

## 四、质量建议

- **数量**：领域微调建议 500–5000 条高质量数据；仅验证流程几十条即可。
- **质量 > 数量**：宁要 500 条精品，不要 5 万条噪声。
- **格式统一**：output 要符合你期望的回复风格（语气、长度、结构）。
- **覆盖边界**：包含"不知道/拒绝"类样本，避免模型乱编。

## 五、示例数据

运行 `python scripts/make_sample_data.py` 可生成约 50 条示例数据，用于跑通全流程。正式训练请替换为你的真实数据。
