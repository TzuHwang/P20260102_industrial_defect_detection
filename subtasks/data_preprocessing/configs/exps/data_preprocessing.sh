set -e
set -x

python -m subtasks.data_preprocessing.scripts.main \
    --task inference \
    --yml-config subtasks/data_preprocessing/configs/yaml/data_preprocessing.yml
