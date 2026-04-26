docker run -dit \
    --name tape_measure \
    -v 'D:\code_library':'/root/code_library' \
    -v 'E:\cache_data':'/root/data' \
    -v 'C:\Users\User\.claude':'/root/.claude':ro \
    --shm-size 256gb \
    --gpus all \
    private_project_template:ci_build bash
