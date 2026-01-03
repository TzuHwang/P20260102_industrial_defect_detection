docker run -dit \
    --name template \
    -v 'D:\code_library':'/root/code_library' \
    -v 'D:\data':'/root/data' \
    --shm-size 256gb \
    --gpus all \
    nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 bash
