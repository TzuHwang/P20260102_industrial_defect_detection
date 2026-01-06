FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

WORKDIR /provate_project_template
COPY . .

# Install system dependencies
RUN apt update && apt install -y \
    curl git wget \
    python3 python3-pip python3-venv \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

# 1. Setup Venv & Path
ENV VIRTUAL_ENV=/venv/default
RUN python3 -m venv $VIRTUAL_ENV
# Add venv bin to path FIRST so 'python' and 'pip' point there automatically
ENV PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"

# 2. Install Poetry
# We use the venv's python directly to avoid system package conflicts
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.0.1

# 3. Poetry Config
# Since we are already in a venv and PATH is set, poetry will use it
RUN poetry config virtualenvs.create false

# 4. Install dependencies
RUN poetry install --no-cache --no-root -vv && poetry run poe install

# 5. Persistent Activation for interactive shells
RUN echo "source $VIRTUAL_ENV/bin/activate" >> /root/.bashrc

CMD ["bash"]