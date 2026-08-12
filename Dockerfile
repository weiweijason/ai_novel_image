# Image Worker - ComfyUI 影像生成
# 硬體需求: RTX 5090 32GB VRAM

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 建立虛擬環境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 模型目錄 (掛載 Volume)
RUN mkdir -p /models/checkpoints /models/vae /models/controlnet

# 環境變數預設值
ENV API_BASE_URL=http://api:8000
ENV WORKER_ID=image-01
ENV WORKER_TYPE=image
ENV WORKER_HOSTNAME=image-worker
ENV COMFYUI_API_URL=http://localhost:8188
ENV S3_ENDPOINT=http://minio:9000
ENV S3_BUCKET=assets
ENV SD_MODEL_PATH=/models/sd/kohaku-v4.1.safetensors

# 禁用 Python 輸出緩衝，讓日誌即時顯示
ENV PYTHONUNBUFFERED=1

# 啟動命令 (使用 -u 參數確保輸出即時顯示)
CMD ["python", "-u", "main.py"]
