# 部署指南

训练产物如何回本地并投入使用。

## 一、导入 Ollama

```bash
# 1. 确认 GGUF 文件在本地
ls outputs/gguf/

# 2. 导入
python scripts/import_ollama.py --gguf outputs/gguf/<你的模型>.gguf --name personal-agi-9b

# 3. 对话验证
ollama run personal-agi-9b
```

## 二、配置推理后端

编辑 `config/config.yaml` 的 `inference` 段：

```yaml
inference:
  backend: "ollama"          # 本地用 ollama
  ollama:
    host: "http://localhost:11434"
    model: "personal-agi-9b" # 导入的模型名
```

## 三、三种使用方式

```bash
# 1. 图形界面（推荐日常用）
uv run streamlit run app/app.py

# 2. 命令行
uv run python scripts/chat.py

# 3. OpenAI 兼容 API（接入其它工具）
uv run uvicorn app.api:app --port 8000
# 然后可用 openai SDK 指向 http://localhost:8000/v1 调用
```

## 四、接入现有工具链（可选）

你的 `personal-agi-9b` 通过 OpenAI 兼容接口，可无缝接入：
- deepseek-web-api（替换模型后端）
- workbench
- 任意支持自定义 base_url 的客户端

## 五、性能参考（本地 RTX 5060 8GB）

- 9B Q4_K_M 量化约 6GB，可完整载入显存。
- 生成速度约 15–30 token/s（视 GPU 而定）。

## 六、更新迭代

1. 收集新数据 → 2. 云端重训 → 3. 重新量化 → 4. `ollama create` 覆盖导入
