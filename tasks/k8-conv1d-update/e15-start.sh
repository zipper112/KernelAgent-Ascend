#!/bin/bash
# E065 复刻 on e15（npu-8）：最新 main f505184be15b + main worktree（E065 patch dbb3b04）
# 与 npu-7 E065 的唯一差异：路径（e15 本地 worktree/模型）+ 容器名 glm-kda + 端口 8001
# 双臂开关：$1 = vec|orig（GLMFLASH_K8_VEC=1|0）；默认 vec
set -e
ARM=${1:-vec}
PORT=${GLM_PORT:-8001}
docker rm -f glm-kda >/dev/null 2>&1 || true
if curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
  echo "[preflight] 端口 $PORT 被占，拒绝启动"; docker ps --format '{{.Names}} {{.Status}}' | head -5; exit 1
fi
K8VEC=$([ "$ARM" = "vec" ] && echo 1 || echo 0)
exec docker run -d --restart unless-stopped --name glm-kda \
  --privileged --net=host --shm-size=500g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/sbin:/usr/local/sbin \
  -v /dev/npu-smi:/dev/npu-smi -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /data01/models/GLM-5.3-Flash-w8a8:/model:ro \
  -v /data02/kda/vllm-ascend-main:/vllm-workspace/vllm-ascend \
  -e GLMFLASH_K8_VEC=$K8VEC \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True -e HCCL_OP_EXPANSION_MODE=AIV -e HCCL_BUFFSIZE=1024 \
  quay.io/ascend/vllm-ascend:nightly-main \
  vllm serve /model --served-model-name GLM-5.3-Flash \
    --tensor-parallel-size 8 --data-parallel-size 1 --enable-expert-parallel \
    --quantization ascend --safetensors-load-strategy prefetch \
    --max-model-len 133120 --max-num-seqs 32 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 --api-server-count 1 --seed 1024 \
    --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45 \
    --enable-prefix-caching --enable-auto-tool-choice --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
    --port $PORT
