FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl bzip2 ca-certificates git \
      libgomp1 libstdc++6 \
      libhdf5-dev libffi-dev \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba and create a Python 3.11 env
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH="/opt/conda/bin:${PATH}"

RUN micromamba create -y -n state python=3.11 -c conda-forge \
    && micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "state", "/bin/bash", "-c"]

# PyTorch with CUDA 12.6
RUN pip install --no-cache-dir \
      torch==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu126

# arc-state from STATE_expansion fork (includes QC cross-attention)
ARG STATE_REF=main
RUN pip install --no-cache-dir \
      "arc-state @ git+https://github.com/ArnovanHilten/STATE_expansion.git@${STATE_REF}"

# Verify key imports
RUN python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" \
 && python -c "import state; print('state OK')" \
 && python -c "import cell_eval; print('cell_eval OK')"

# Wrapper commands
RUN printf '%s\n' '#!/usr/bin/env bash' 'exec micromamba run -n state state "$@"' > /usr/local/bin/state \
 && chmod +x /usr/local/bin/state \
 && printf '%s\n' '#!/usr/bin/env bash' 'exec micromamba run -n state cell-eval "$@"' > /usr/local/bin/cell-eval \
 && chmod +x /usr/local/bin/cell-eval

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["micromamba", "run", "-n", "state", "bash"]
