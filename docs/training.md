# 云端训练指南（AutoDL）

在 AutoDL 上微调 Qwen3.5-9B。

## 一、为什么必须用云端

9B 模型 QLoRA 微调需约 10GB+ 显存，本地 8GB 显卡（如 RTX 5060）无法承载。AutoDL 提供 RTX 4090（24GB）/ A100（40/80GB），可轻松训练 9B。

## 二、开卡

1. 登录 [AutoDL](https://www.autodl.com)，充值少量金额（几元即可）。
2. 创建实例：
   - **显卡**：RTX 4090（24GB，推荐）或 A100
   - **镜像**：PyTorch 2.4 + CUDA 12.1（官方镜像）
   - **数据盘**：扩容到 50GB+（9B 权重 + 训练产物约 30GB）
3. 启动实例，通过 SSH 或 JupyterLab 进入。

## 三、上传项目与数据

```bash
# 方式 A：git 上传（推荐）
git clone https://github.com/<你的用户名>/personal-agi.git
cd personal-agi

# 方式 B：scp 上传本地目录
scp -r -P <端口> ./personal-agi root@<实例地址>:/root/autodl-tmp/
```

把训练数据放到 `data/processed/train.jsonl` 与 `eval.jsonl`。

## 四、安装依赖

```bash
cd /root/autodl-tmp/personal-agi
pip install -r docker/requirements-train.txt
# 若 unsloth 安装报版本不匹配，按 Unsloth 官方说明调整 CUDA/PyTorch 版本
```

## 五、训练

```bash
python scripts/train_cloud.py --config config/config.yaml
```

训练参数在 `config/config.yaml` 的 `training` 段可调（轮数、学习率、LoRA rank 等）。

产物输出到 `outputs/<run_name>/lora_adapter/`。

## 六、合并 + 量化

```bash
python scripts/merge.py --config config/config.yaml --adapter outputs/<run_name>/lora_adapter
python scripts/quantize.py --config config/config.yaml --model outputs/merged
```

量化产物在 `outputs/gguf/`，约 6GB 的 GGUF 文件。

## 七、下载回本地

```bash
# 本地执行
scp -r -P <端口> root@<实例地址>:/root/autodl-tmp/personal-agi/outputs/gguf ./outputs/
```

## 八、成本参考

- RTX 4090：约 ¥1.5–2.5/小时
- 一个几千条数据的 9B QLoRA 微调（2–3 epoch）：约 1–3 小时，总花费几元到十几元
- **训练完成后务必及时关机/释放实例，避免持续计费**

## 九、注意事项

- 训练前先 `nvidia-smi` 确认显存。
- 首次会下载 9B 基座权重（约 18GB），耗时取决于网络。
- 数据量小（<100 条）时，调小 `num_epochs` 或增大 `eval_steps`，避免过拟合。
