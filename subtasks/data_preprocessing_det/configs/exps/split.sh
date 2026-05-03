set -e
set -x

python -m subtasks.data_preprocessing_det.scripts.split \
    --yml-config subtasks/data_preprocessing_det/configs/yaml/data_split_back.yml
