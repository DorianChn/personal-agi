.PHONY: setup data train merge quantize import ui api chat clean

# 建环境
setup:
	uv sync

# 生成示例数据
data:
	uv run python scripts/make_sample_data.py

# 数据清洗（需先放数据到 data/raw）
clean:
	uv run python scripts/clean.py --input data/raw --output data/processed

# 云端训练（在 AutoDL 上执行）
train:
	python scripts/train_cloud.py --config config/config.yaml

# 合并 LoRA
merge:
	python scripts/merge.py --config config/config.yaml --adapter outputs/run/lora_adapter

# 量化 GGUF
quantize:
	python scripts/quantize.py --config config/config.yaml --model outputs/merged

# 导入 Ollama
import:
	python scripts/import_ollama.py --gguf outputs/gguf/*.gguf --name personal-agi-9b

# 启动 UI
ui:
	uv run streamlit run app/app.py

# 启动 API
api:
	uv run uvicorn app.api:app --port 8000

# CLI 对话
chat:
	uv run python scripts/chat.py
