from nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

apt update && apt install -y \
    curl \
    python3 python3-pip python3-venv \
    xvfb 


# setup default venv with poetry
## make venv dir
mkdir -p /venv/default
python3 -m venv /venv/default
source /venv/default/bin/activate


## install poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# set poetry config
poetry config virtualenvs.path /venv
poetry config virtualenvs.create false

# install dependencies
poetry install --no-cache -vv && poetry run poe install
