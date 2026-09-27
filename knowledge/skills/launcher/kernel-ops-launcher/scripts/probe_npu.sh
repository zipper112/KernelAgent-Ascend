#!/bin/bash
# probe_npu.sh —— 阶段②NPU 机器探查（只读优先；容器验证用 --rm 一次性容器）
# 用法：bash probe_npu.sh --host <host> [--jump <jump>] [--device <id>] [--image <img>] [--verify]
#   --verify = 校验模式（只验连通+卡在，不重探容器）
# 产出：stdout JSON（host/arch/device 建议/镜像结论/卡清单）
set -uo pipefail

HOST=""; JUMP=""; DEVICE=""; IMAGE=""; VERIFY_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --jump) JUMP="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --verify) VERIFY_ONLY=1; shift ;;
    *) shift ;;
  esac
done
[ -z "$HOST" ] && { echo '{"error": "need --host"}'; exit 2; }

# ssh 包装：有 jump 则两跳，否则直连
rsh() {
  if [ -n "$JUMP" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$JUMP" "timeout 60 ssh -o BatchMode=yes $HOST $1"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "timeout 60 $1"
  fi
}

# --- 连通 ---
if ! rsh "echo ok" >/dev/null 2>&1; then
  echo "{\"error\": \"unreachable: ${JUMP:+$JUMP ->}$HOST（检查免密/地址）\"}"
  exit 1
fi

# --- npu-smi 卡清单 ---
SMI=$(rsh "npu-smi info" 2>/dev/null)
if [ -z "$SMI" ]; then
  echo '{"error": "npu-smi 不可用（是 NPU 机吗？）"}'
  exit 1
fi
# 卡行样例：| 0  910B4-1  | OK | ... ；统计型号与每卡进程数
MODELS=$(echo "$SMI" | grep -oE '[0-9]{3}[AB][0-9]?[a-zA-Z0-9.-]*' | head -1)
N_CARDS=$(echo "$SMI" | grep -cE '^\| [0-9]+ ')
BUSY=$(echo "$SMI" | grep -c "Process id" || true)

# --- arch 推断（型号→dav 代际）---
arch_of() {
  case "$1" in
    910B4*|910B3*|910B2*) echo dav_2201 ;;
    910B*) echo dav_200 ;;
    910A*) echo dav_100 ;;
    310P*) echo dav_200i ;;
    *) echo "unknown(${1})" ;;
  esac
}
ARCH=$(arch_of "$MODELS")

# --- 空闲卡建议（每卡进程数=0 的第一张；粗粒度：npu-smi 块内无 Process id）---
SUGGEST="${DEVICE:-}"
if [ -z "$SUGGEST" ]; then
  for i in $(seq 0 $((N_CARDS>0 ? N_CARDS-1 : 0))); do
    BLK=$(echo "$SMI" | grep -A6 "NPU $i ")
    if ! echo "$BLK" | grep -q "Process id"; then SUGGEST="$i"; break; fi
  done
  [ -z "$SUGGEST" ] && SUGGEST=0
fi

if [ "$VERIFY_ONLY" = "1" ]; then
  echo "{\"host\": \"$HOST\", \"reachable\": true, \"arch\": \"$ARCH\", \"device\": \"$SUGGEST\", \"mode\": \"verify-only\"}"
  exit 0
fi

# --- 容器盘点 ---
IMAGES=$(rsh "sudo docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | head -20")

# --- 容器可用性验证（--rm 一次性；torch_npu import + 设备自检）---
CAND=""
if [ -n "$IMAGE" ]; then
  CAND="$IMAGE"
else
  # 优先常见可用镜像（踩坑经验排序：vllm-ascend nightly 系含 triton；xllm 系 CANN 完整）
  for c in $(echo "$IMAGES"); do
    case "$c" in
      *vllm-ascend*|*xllm*|*ascend*triton*) CAND="$c"; break ;;
    esac
  done
fi
CONTAINER_OK=false; CONTAINER_ERR=""
if [ -n "$CAND" ]; then
  OUT=$(rsh "sudo docker run --rm --device /dev/davinci$SUGGEST --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver_host:ro \
    --entrypoint bash $CAND -c \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; \
     export LD_LIBRARY_PATH=/usr/local/Ascend/driver_host/lib64/common:/usr/local/Ascend/driver_host/lib64/driver:\$LD_LIBRARY_PATH; \
     python3 -c \"import torch,torch_npu; print(\\\"NPU-CONTAINER-OK\\\")\"' 2>&1 | tail -3")
  echo "$OUT" | grep -q "NPU-CONTAINER-OK" && CONTAINER_OK=true || CONTAINER_ERR=$(echo "$OUT" | tail -1 | cut -c1-160)
fi

# --- 输出 ---
IM_JSON=$(echo "$IMAGES" | awk '{printf "\"%s\",", $0}' | sed 's/,$//')
cat << EOF
{
  "host": "$HOST",
  "jump": "${JUMP:-null}",
  "reachable": true,
  "model": "${MODELS:-unknown}",
  "n_cards": ${N_CARDS:-0},
  "busy_cards": ${BUSY:-0},
  "arch": "$ARCH",
  "device_suggest": "$SUGGEST",
  "images": [${IM_JSON}],
  "container_image": "${CAND:-null}",
  "container_ok": $CONTAINER_OK,
  "container_err": "${CONTAINER_ERR//\"/\\\"}"
}
EOF
[ "$CONTAINER_OK" = "true" ] && exit 0 || exit 1
