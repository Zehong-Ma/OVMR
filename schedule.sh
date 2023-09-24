#!/bin/bash

while true; do
    # 使用nvidia-smi获取显卡使用情况，提取空闲显存信息（第1块显卡）
    gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=2)
    free_memory=$(echo $gpu_info | awk '{print $1}')

    # 设置一个空闲显存阈值，例如，大于等于2000 MiB视为空闲
    threshold=18000

    if [ "$free_memory" -ge "$threshold" ]; then
        echo "GPU is idle. Running Python script..."
        #python main.py
        sh scripts/train_vit_b-32.sh
        break  # 如果运行成功，退出循环
    else
        echo "GPU is busy. Waiting..."
        sleep 60  # 等待一分钟后继续循环
    fi
done
