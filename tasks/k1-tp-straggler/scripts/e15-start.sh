#!/bin/bash
# K1 三臂启动器：E065 形态（vec 常开）+ GLM_EXTRA_CFG 注入 --additional-config（单变量刀）
# 臂定义（round-0-contract）：c001=空；c002=enable_reduce_sample；c003=两刀组合
set -e
ARM=${1:-vec}
PORT=${GLM_PORT:-8001}
EXTRA=${GLM_EXTRA_CFG:-}
docker rm -f glm-kda >/dev/null 2>&1 || true
if curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
  echo "[preflight] 端口 $PORT 被占，拒绝启动"; exit 1
fi
K8VEC=$([ "$ARM" = "vec" ] && echo 1 || echo 0)
CMD=(vllm serve /model --served-model-name GLM-5.3-Flash \
  --tensor-parallel-size 8 --data-parallel-size 1 --enable-expert-parallel \
  --quantization ascend --safetensors-load-strategy prefetch \
  --max-model-len 133120 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 --api-server-count 1 --seed 1024 \
  --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-prefix-caching --enable-auto-tool-choice --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
  --port $PORT)
if [ -n "$EXTRA" ]; then
  CMD+=(--additional-config "$EXTRA")
fi
exec docker run -d --restart unless-stopped --name glm-kda \
  --privileged --net=host --shm-size=500g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/sbin:/usr/local/sbin \
  -v /dev/npu-smi:/dev/npu-smi -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /data01/models/GLM-5.3-Flash-w8a8:/model:ro \
  -v /data02/kda/vllm-ascend-main:/vllm-workspace/vllm-ascend \
  -e GLMFLASH_K8_VEC=$K8VEC \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True -e HCCL_OP_EXPANSION_MODE=AIV -e HCCL_BUFFSIZE=1024 \
  quay.io/ascend/vllm-ascend:nightly-main "${CMD[@]}"
