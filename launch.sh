#!/bin/bash
# launch.sh —— 战役发射器（比赛形态：一次性注入 phase prompt，codex 自主迭代）
# 用法：bash launch.sh [战役名，默认 b5] [--rounds N 透传给 codex 的时间预算提示]
set -euo pipefail
cd "$(dirname "$0")/.."   # 仓根：/data01/mahaolong/KAgent

BATTLE="${1:-b5}"
LOG="run-logs/codex-${BATTLE}.log"
mkdir -p run-logs

# GLM key（不入仓）
export GLM_API_KEY
GLM_API_KEY=$(python3 -c "
import re
for line in open('agent-config/local-secrets.yaml', encoding='utf-8'):
    m = re.search(r'api_key:\s*\"?([A-Za-z0-9._-]+)', line)
    if m: print(m.group(1)); break
")

TASK_DIR="tasks/k8-triton"
echo "[$(date '+%F %T')] battle ${BATTLE}: codex exec -> ${TASK_DIR} (log ${LOG})"

exec codex exec \
  -s danger-full-access \
  --skip-git-repo-check \
  -C "${TASK_DIR}" \
  - "$(cat ${TASK_DIR}/phase.md)" 2>&1 | tee -a "${LOG}"
