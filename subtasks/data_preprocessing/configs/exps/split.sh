set -e
set -x

python -m subtasks.data_preprocessing.scripts.split \
    --yml-config subtasks/data_preprocessing/configs/yaml/data_split_front.yml
