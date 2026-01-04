docker run -dit \
    --name template_view \
    -v 'D:\code_library':'/root/code_library' \
    -v 'D:\data':'/root/data' \
    --shm-size 256gb \
    --gpus all \
    tzuhwang/nous:0.1.0 bash
