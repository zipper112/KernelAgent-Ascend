#!/bin/bash
# K1 取证配置：vec 基线（无参数刀）+ torch profiler 端点（E057 同款 --profiler-config）
# 用法：bash e15-start-prof.sh ；就绪后 POST /start_profile → 跑负载 → /stop_profile → 拉 /prof
set -e
PORT=${GLM_PORT:-8001}
docker rm -f glm-kda >/dev/null 2>&1 || true
mkdir -p /data02/kda/prof
exec docker run -d --restart unless-stopped --name glm-kda \
  --privileged --net=host --shm-size=500g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/sbin:/usr/local/sbin \
  -v /dev/npu-smi:/dev/npu-smi -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /data01/models/GLM-5.3-Flash-w8a8:/model:ro \
  -v /data02/kda/vllm-ascend-main:/vllm-workspace/vllm-ascend \
  -v /data02/kda/prof:/prof \
  -e GLMFLASH_K8_VEC=1 \
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
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/prof"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
    --port $PORT
