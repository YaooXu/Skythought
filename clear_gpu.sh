#!/bin/bash

echo "Killing all processes using GPU..."

# 获取所有使用 GPU 的用户进程的 PID（排除 PID 为空的行）
PIDS=$(nvidia-smi | grep -E "[0-9]+ +[0-9]+ +[0-9]+" | awk '{print $5}' | sort -u)

if [ -z "$PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

echo "Found GPU processes with PIDs: $PIDS"

# 遍历 PID 并杀死
for pid in $PIDS; do
    echo "Killing PID $pid"
    kill -9 $pid
done

echo "All GPU processes killed."
