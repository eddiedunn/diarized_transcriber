FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Install gpu-common wheel
COPY deploy/gpu_common-0.1.0-py3-none-any.whl /tmp/
RUN uv pip install --system --python python3.11 /tmp/gpu_common-0.1.0-py3-none-any.whl \
    && rm /tmp/gpu_common-0.1.0-py3-none-any.whl

# Install diarized_transcriber
WORKDIR /app
COPY . /src/diarized_transcriber
RUN uv pip install --system --python python3.11 "/src/diarized_transcriber[server,profiles]"

# Environment defaults
ENV DIARIZED_HOST=0.0.0.0
ENV DIARIZED_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3.11", "-m", "diarized_transcriber.api.server"]
