# private_project_template
A template design for private machine learning projects.  
Maintainer: Tzu-Hsiang Wang  
Init-date: 2026/01/03

---
## Project Overview
This repository provides a **starter template** for private ML projects with GPU support, reproducible environments using **Poetry**, and a focus on PyTorch-based pipelines.

Key features:

- GPU-ready development environment
- Python >= 3.12
- Dependency management with [Poetry](https://python-poetry.org/)
- Optional ML frameworks: PyTorch, mmengine, mmpretrain, mmdet, mmsegmentation
- Predefined tasks via [Poe](https://github.com/nat-n/poethepoet)

---
##  Directory Structure

Project-folder
    ├── configs/ # Configuration files (YAML, JSON) for experiments
    ├── project_src/ # Source code
    │ ├── init.py
    │ └── main.py # Entry point
    ├── subtasks/ # Sub-modules / pipeline tasks
    ├── tests/ # Unit tests
    │ └── test_main.py
    ├── workflow/ # Scripts to initialize container, setup environment
    ├── Dockerfile # Docker image for dev environment
    ├── .flake8 # Code style config
    ├── .gitignore
    ├── pyproject.toml # Poetry project config
    ├── poetry.lock # Locked dependencies
    └── README.md

## build docker image and init container

*build docker image*
`configs/docker/build.sh`


*init container*
`configs/docker/init_container.sh`

---
## Run test 
`pytest -vv`

---