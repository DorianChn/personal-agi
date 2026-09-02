# AutoDL 云端训练镜像
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /root/autodl-tmp/personal-agi

# 安装训练依赖
COPY docker/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# 复制项目
COPY . .

CMD ["python", "scripts/train_cloud.py", "--config", "config/config.yaml"]
