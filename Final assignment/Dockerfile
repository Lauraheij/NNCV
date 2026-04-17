FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# install dependencies (add any additional dependencies your project requires)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    pillow \
    torchmetrics \
    timm 

WORKDIR /app

# We put them here so changing them doesn't trigger a pip re-install
ARG PREDICT_FILE=predict.py
ARG MODEL_ARCH_FILE=model.py
ARG MODEL_WEIGHTS_FILE=model.pt
ARG PRETRAIN_PTH=dinov2_vitb14_pretrain.pth
ARG PRETRAIN_PTH_DEST

# Use the variables to copy files into fixed names for the server
COPY ${PREDICT_FILE} /app/predict.py
COPY ${MODEL_ARCH_FILE} /app/model.py
COPY ${MODEL_WEIGHTS_FILE} /app/model.pt
COPY ${PRETRAIN_PTH} /app/${PRETRAIN_PTH_DEST}
COPY dinov2_hub /app/dinov2_hub

ENTRYPOINT ["python", "/app/predict.py"]