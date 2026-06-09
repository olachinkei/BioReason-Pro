FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.8.17
ARG INSTALL_FLASH_ATTN=false
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG ESM_VERSION=3.2.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    BIOREASON_RUNTIME_ROOT=/mnt/data/bioreason/BioReason-Pro \
    TORCHINDUCTOR_COMPILE_THREADS=4 \
    VLLM_ATTENTION_BACKEND=XFORMERS \
    VLLM_USE_V1=false \
    BIOREASON_VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libaio-dev \
    rsync \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
RUN uv python install 3.11

WORKDIR /workspace/BioReason-Pro

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

RUN python -m pip install --no-deps "esm==${ESM_VERSION}" \
    && if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
        MAX_JOBS=8 python -m pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"; \
    fi

COPY . .
RUN uv sync --frozen

RUN chmod +x docker/entrypoint.sh

ENTRYPOINT ["/workspace/BioReason-Pro/docker/entrypoint.sh"]
CMD ["python", "train.py", "--help"]
