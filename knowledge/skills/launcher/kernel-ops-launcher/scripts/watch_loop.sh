#!/bin/bash
# watch_loop.sh —— 阶段⑤战役监听（root agent 每 10-15min 调一次）
# 用法：bash watch_loop.sh <repo> <战役名>
# 输出 JSON：进程/日志增量/内环次数/账本行/git commit/终局判定/卡死判定
set -uo pipefail
REPO="${1:?repo}"; B="${2:?battle}"
LOG="$REPO/run-logs/codex-$B.log"
STATE="$REPO/run-logs/.watch-$B.state"

if [ ! -f "$LOG" ]; then echo "{\"error\": \"no log: $LOG\"}"; exit 2; fi

# 进程存活
ALIVE=$(pgrep -f "codex exec" >/dev/null 2>&1 && echo true || echo false)

# log 尺寸/增量（对照上次快照）
SZ=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
LAST_SZ=0; [ -f "$STATE" ] && LAST_SZ=$(cat "$STATE" 2>/dev/null | head -1)
echo "$SZ" > "$STATE"
GROW=$((SZ - LAST_SZ))

# 行为统计（累计）
V_FAST=$(grep -c "verify.py.*--fast" "$LOG" 2>/dev/null || echo 0)
V_ALL=$(grep -c "python verify.py\|python3 verify.py" "$LOG" 2>/dev/null || echo 0)
QUERY=$(grep -c "router/query.py" "$LOG" 2>/dev/null || echo 0)
SKILL=$(grep -c "SKILL.md" "$LOG" 2>/dev/null || echo 0)
BENCH=$(grep -c "bench.py --record" "$LOG" 2>/dev/null || echo 0)

# 账本与 git
CSV="$REPO/tasks/k8-triton/docs/benchmark.csv"   # TODO: 任务名参数化（多任务时由 root agent 传路径）
CSV_ROWS=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )
GIT_NEW=$(git -C "$REPO" log --oneline -1 2>/dev/null | head -1)

# 终局/卡死判定（log 尾 40 行）
TAIL=$(tail -40 "$LOG" | grep -v "^warning" | tail -8)
if echo "$TAIL" | grep -qE "终局|达标|完成判据|verdict"; then
  STATUS="finished?"
elif [ "$GROW" -lt 100 ] && [ "$ALIVE" = "true" ]; then
  STATUS="stalled?(15min 无增长需人工看)"
else
  STATUS="running"
fi

# 最新 w02 数字（若有）
LAST_MEAN=$(grep -E "^\s+w[0-9]+: mean=" "$LOG" 2>/dev/null | tail -1 | tr -d ' ')

cat << EOF
{
  "battle": "$B",
  "alive": $ALIVE,
  "log_bytes": $SZ,
  "log_growth": $GROW,
  "verify_fast": $V_FAST,
  "verify_total": $V_ALL,
  "router_queries": $QUERY,
  "skill_reads": $SKILL,
  "bench_recorded": $BENCH,
  "ledger_rows": $CSV_ROWS,
  "git_head": "${GIT_NEW//\"/\\\"}",
  "last_mean": "${LAST_MEAN//\"/\\\"}",
  "status": "$STATUS",
  "tail_hint": "$(echo "$TAIL" | tail -2 | head -1 | cut -c1-140 | tr '\n' ' ' | sed 's/"/\\"/g')"
}
EOF
