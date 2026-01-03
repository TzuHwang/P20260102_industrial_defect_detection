from nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

WORKDIR /provate_project_template
COPY . .

RUN apt update && apt install -y \ 
        curl git wget \
        python3 python3-pip python3-venv \
        xvfb 


# setup default venv with poetry
## make venv dir
RUN mkdir -p /venv/default
RUN python3 -m venv /venv/default
RUN source /venv/default/bin/activate


## install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
RUN source ~/.bashrc

# set poetry config
RUN poetry config virtualenvs.path /venv
RUN poetry config virtualenvs.create false

# install dependencies
RUN poetry install --no-cache -vv && poetry run poe install
