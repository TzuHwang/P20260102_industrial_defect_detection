docker run -dit \
    --name template_view \
    -v 'D:\code_library':'/root/code_library' \
    -v 'D:\data':'/root/data' \
    --shm-size 256gb \
    --gpus all \
    private_project_template:ci_build bash
