set -e
set -x

python -m subtasks.label_wash.scripts.main \
    --task inference \
    --yml-config subtasks/label_wash/configs/yaml/default.yml
