set -e
set -x

python -m subtasks.data_preprocessing_det.scripts.main \
    --task inference \
    --yml-config subtasks/data_preprocessing_det/configs/yaml/data_preprocessing.yml
