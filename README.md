# Personal AGI — 你的专属 9B 模型训练与部署

一套完整的个人 AI 助手工程：**数据 → 微调 → 量化 → 部署 → 日常使用** 全流程闭环。

以 `Qwen3.5-9B` 为基座，在云端 GPU（AutoDL）做 LoRA/QLoRA 微调，产出**属于你自己的专属模型**，再量化成 GGUF 回本地通过 Ollama 推理，最终接入一套带工具调用、RAG、记忆的 Agent 界面。

> 硬件说明：本项目针对 **8GB 显存本地机**（如 RTX 5060）+ **云端 GPU（AutoDL RTX 4090/A100）** 的混合架构。9B 模型无法在 8GB 显存上微调（QLoRA 也需约 10GB+），故训练放云端，推理回本地。

---

## 整体架构

```
┌──────────────────────────────────────────────────────┐
│  使用层 Interface                                     │
│  Streamlit Chat UI  ·  CLI  ·  FastAPI (OpenAI 兼容)  │
└───────────────┬──────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────┐
│  编排层 Agent Core（可选扩展）                          │
│  对话管理 · 工具调用 · 记忆 · RAG 检索                  │
└───────────────┬──────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────┐
│  推理层 Inference                                     │
│  Ollama（本地 Qwen3.5-9B）── 云端 API（可切换）         │
└───────────────┬──────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────┐
│  模型资产层 Model Assets                               │
│  基座 → 微调(LoRA) → 合并 → GGUF 量化 → 导入 Ollama     │
└──────────────────────────────────────────────────────┘
```

## 技术选型

| 环节 | 选型 | 说明 |
|---|---|---|
| 基座模型 | Qwen3.5-9B（HuggingFace 权重） | 云端 QLoRA 微调 |
| 微调框架 | Unsloth + LoRA/QLoRA | 显存占用低、速度快 |
| 量化 | GGUF (Q4_K_M) + llama.cpp | 回本地约 6GB，Ollama 直跑 |
| 推理 | Ollama | OpenAI 兼容 API |
| Agent | 自研轻量编排 + 工具调用 + RAG | bge-m3 向量化 |
| 界面 | Streamlit + FastAPI | 本地 UI + 可脚本化 API |
| 环境 | uv + pyproject.toml | 快速、可复现 |

## 目录结构

```
personal-agi/
├── README.md                 # 本文件
├── LICENSE                   # MIT
├── pyproject.toml            # uv 依赖
├── .gitignore
├── Makefile                  # 一键命令入口
├── config/config.yaml        # 训练/推理参数
├── data/
│   ├── raw/                  # 原始语料
│   ├── processed/            # 清洗后训练数据
│   └── samples/              # 示例数据
├── scripts/
│   ├── make_sample_data.py   # 生成示例训练数据
│   ├── clean.py              # 数据清洗
│   ├── train_cloud.py        # 云端微调（Unsloth QLoRA）
│   ├── merge.py              # LoRA 合并
│   ├── quantize.py           # GGUF 量化
│   ├── import_ollama.py      # 导入 Ollama
│   └── chat.py               # CLI 对话
├── src/
│   ├── agent/                # 编排层
│   ├── inference/            # 推理客户端
│   └── utils/
├── app/
│   ├── app.py                # Streamlit UI
│   └── api.py                # FastAPI 服务
├── docker/                   # 云端训练镜像
└── docs/                     # 详细文档
```

## 快速开始

### 0. 前置要求

- Python 3.10+（推荐 3.11）、[uv](https://docs.astral.sh/uv/)
- 本地已装 [Ollama](https://ollama.com)（用于推理）
- 云端：AutoDL 账号 + 一张 RTX 4090/A100（用于训练 9B）

### 1. 建环境

```bash
git clone https://github.com/<你的用户名>/personal-agi.git
cd personal-agi
uv sync
```

### 2. 生成示例数据（演示全流程）

```bash
uv run python scripts/make_sample_data.py
# 产出 data/processed/train.jsonl 与 eval.jsonl（约 50 条示例问答）
```

### 3. 本地训练小模型（可选，验证流程）

8GB 显存可训 0.6B / 1.7B，详见 [docs/training-local.md](docs/training-local.md)：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy transformers peft trl bitsandbytes datasets accelerate
python scripts/train_local.py --model Qwen/Qwen3-0.6B --epochs 3
python scripts/train_local.py --model Qwen/Qwen3-1.7B --epochs 2
```

### 4. 云端训练 9B（AutoDL）

详见 [docs/training.md](docs/training.md)，核心命令：

```bash
# 在 AutoDL 实例内：
cd /root/autodl-tmp/personal-agi
pip install -r docker/requirements-train.txt
python scripts/train_cloud.py --config config/config.yaml
```

### 5. 合并 + 量化

```bash
python scripts/merge.py --config config/config.yaml
python scripts/quantize.py --config config/config.yaml
```

### 6. 导入 Ollama + 本地推理

```bash
python scripts/import_ollama.py --name personal-agi-9b
ollama run personal-agi-9b
```

### 7. 日常使用

```bash
uv run streamlit run app/app.py     # 打开浏览器对话界面
uv run python scripts/chat.py        # 或命令行对话
uv run uvicorn app.api:app --port 8000   # 或 OpenAI 兼容 API
```

## 文档

- [架构设计](docs/architecture.md)
- [数据收集与预处理](docs/data.md)
- [云端训练指南](docs/training.md)
- [部署指南](docs/deployment.md)

## License

[MIT](LICENSE)

---

## 相关仓库

- [qwen3-8-9b-scz](https://github.com/DorianChn/qwen3-8-9b-scz) — 本流水线产出的专属模型发布仓（GGUF 与评测结果）
