#!/bin/bash
# e15 上的双臂压测驱动（由 crontab @reboot 式一次性触发或手动 bash 调用，绝对脱离 ssh 会话）
# 用法：bash e15-run-bench.sh <arm-tag>   （arm-tag 仅入结果文件名，如 vec-r2 / orig-r2）
TAG=${1:?need-tag}
cd /data02/kda
exec python3 e15-bench.py 8001 "$TAG" > "/data02/kda/${TAG}.json" 2> "/data02/kda/${TAG}.err"
