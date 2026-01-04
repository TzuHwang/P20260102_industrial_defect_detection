FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

WORKDIR /provate_project_template
COPY . .

# Install system dependencies
RUN apt update && apt install -y \ 
    curl git wget \
    python3 python3-pip python3-venv \
    xvfb && \
    rm -rf /var/lib/apt/lists/*


# setup default venv with poetry
## 1. Setup default venv
RUN mkdir -p /venv/default
RUN python3 -m venv /venv/default
ENV PATH="/venv/default/bin:/root/.local/bin:$PATH" \
    PIP_BREAK_SYSTEM_PACKAGES=1

## 2. Install poetry using the venv's python
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.0.1

## 3. Set poetry config
RUN poetry config virtualenvs.path /venv
RUN poetry config virtualenvs.create false

## 4. install dependencies
RUN poetry install --no-cache --no-root -vv && poetry run poe install
