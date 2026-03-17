docker run -dit \
    --name template_view \
    -v 'D:\code_library':'/root/code_library' \
    -v 'D:\data':'/root/data' \
    --shm-size 256gb \
    --gpus all \
    P20260102_industrial_defect_detection:ci_build bash
