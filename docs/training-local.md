# 本地训练指南（8GB 显存）

在本地 RTX 5060（8GB 显存）上微调小模型，验证全流程。

## 一、能训什么

| 模型 | 显存占用 | 结论 |
|---|---|---|
| Qwen3-0.6B | ~2-3GB | ✅ 轻松 |
| Qwen3-1.7B | ~6-8GB | ⚠️ 较紧，需梯度检查点 |
| Qwen3-4B+ | >8GB | ❌ 不可 |

本地训练的定位：**验证「数据→训练→部署」全流程跑通**，以及在小模型上快速迭代你的数据/提示词。最终 9B 专属模型仍需云端（见 training.md）。

## 二、安装训练依赖

```bash
# 安装 CUDA 版 torch（按你的 GPU 选 CUDA 版本，RTX 50 系用 cu128）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 其余训练依赖
pip install numpy transformers peft trl bitsandbytes datasets accelerate modelscope
```

## 三、国内网络：用 Modelscope 下载模型

HuggingFace 在国内直连不稳定，推荐用 Modelscope（阿里云镜像）下载基座：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', cache_dir='D:/models')"
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', cache_dir='D:/models')"
```

## 四、训练

```bash
# 0.6B（推荐先跑这个，用 Modelscope 本地路径）
python scripts/train_local.py --model D:/models/models/Qwen--Qwen3-0.6B/snapshots/master --epochs 3

# 1.7B（显存较紧，已自动降 batch size 和 seq length）
python scripts/train_local.py --model D:/models/models/Qwen--Qwen3-1.7B/snapshots/master --epochs 2
```

> 若网络能直连 HF，也可直接用 `--model Qwen/Qwen3-0.6B`（脚本已内置 hf-mirror 回退）。

产物输出到 `outputs/local/<模型名-时间戳>/lora_adapter/`。

## 五、显存优化技巧（8GB）

- 已默认开启：梯度检查点 + fp16
- 1.7B 若 OOM：进一步减小 `max_seq_length`（如 256）或 `lora_r`
- 训练前关闭占用显存的程序（如其它正在跑的模型）

## 六、训练完成后

LoRA adapter 可：
1. 直接合并推理（`scripts/merge.py`，需改用本地基座路径）
2. 或先验证效果，满意后再迁移到云端训 9B

## 七、与云端的关系

```
本地 0.6B/1.7B  ──→ 快速验证数据质量、提示词、流程
                       │
                       ▼ 确认无误后
云端 9B（AutoDL） ──→ 正式训练你的专属大模型
```
