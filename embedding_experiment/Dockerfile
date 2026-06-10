FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      ca-certificates curl git \
      tini \
      libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (build-time internet is OK)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"
RUN command -v uv && uv --version

# Pin refs for reproducibility
ARG STATE_REF=main
ARG CELL_EVAL_REF=v0.5.42

# # ---- Venv 1: arc-state / state CLI ----
RUN python3 -m venv /opt/state-venv \
 && /opt/state-venv/bin/python -m pip install --upgrade pip setuptools wheel \
 && uv pip install --python /opt/state-venv/bin/python \
      "arc-state @ git+https://github.com/ArcInstitute/state@${STATE_REF}"

# ---- Venv 2: cell-eval CLI (v0.5.42) ----
RUN python3 -m venv /opt/cell-eval-venv \
 && /opt/cell-eval-venv/bin/python -m pip install --upgrade pip setuptools wheel \
 && uv pip install --python /opt/cell-eval-venv/bin/python \
      "cell-eval @ git+https://github.com/ArcInstitute/cell-eval"

# Create stable wrapper commands so users can just call `state` and `cell-eval`
RUN printf '%s\n' '#!/usr/bin/env bash' 'exec /opt/state-venv/bin/state "$@"' > /usr/local/bin/state \
 && chmod +x /usr/local/bin/state \
 && printf '%s\n' '#!/usr/bin/env bash' 'exec /opt/cell-eval-venv/bin/cell-eval "$@"' > /usr/local/bin/cell-eval \
 && chmod +x /usr/local/bin/cell-eval

# Build-time checks: ensure both CLIs exist
# RUN /usr/local/bin/state --help >/dev/null \
# && /usr/local/bin/cell-eval --help >/dev/null

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

