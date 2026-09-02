# 架构设计

## 一、分层架构

本项目采用四层架构，核心原则是**训练与推理解耦**：

```
┌─────────────── 使用层 ───────────────┐
│  Streamlit UI · CLI · FastAPI         │
├─────────────── 编排层 ───────────────┤
│  对话管理 · 工具调用 · 记忆 · RAG      │
├─────────────── 推理层 ───────────────┤
│  Ollama（本地） ↔ OpenAI 兼容 API     │
├─────────────── 模型资产层 ───────────┤
│  基座 → LoRA → 合并 → GGUF → 导入     │
└──────────────────────────────────────┘
```

## 二、各层职责

### 1. 使用层（Interface）
- `app/app.py`：Streamlit 图形界面，本地浏览器交互。
- `scripts/chat.py`：命令行交互。
- `app/api.py`：FastAPI，提供 OpenAI 兼容 `/v1/chat/completions`，便于接入其它工具链（如你现有的 workbench、deepseek-web-api）。

### 2. 编排层（Agent Core）
预留的智能体能力（`src/agent/`），按需扩展：
- **对话管理**：多轮上下文维护。
- **工具调用**：Function Calling，让模型调用外部工具。
- **记忆**：短期会话记忆 + 长期知识沉淀。
- **RAG**：用 `bge-m3` 向量化私有知识库，检索增强生成。

### 3. 推理层（Inference）
- 统一客户端 `src/inference/client.py`，封装 Ollama 与 OpenAI 两种后端，通过 `config.yaml` 切换。
- 本地用 Ollama 跑量化后的 `personal-agi-9b`。

### 4. 模型资产层（Model Assets）
训练流水线：
```
基座 Qwen3.5-9B (HF)
  → QLoRA 微调（train_cloud.py，云端）
  → LoRA adapter
  → 合并（merge.py）
  → GGUF 量化（quantize.py）
  → 导入 Ollama（import_ollama.py）
```

## 三、关键技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 训练设备 | AutoDL 云端 GPU | 9B 无法在 8GB 本地显存微调 |
| 微调方法 | QLoRA（4bit + LoRA） | 24GB 显存可训 9B，成本低 |
| 量化 | GGUF Q4_K_M | 体积约 6GB，本地 8GB 可全量载入 |
| 推理 | Ollama | 已安装，OpenAI 兼容，生态成熟 |
