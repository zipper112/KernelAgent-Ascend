#!/usr/bin/env bash
# deps/setup.sh —— 自包含校验（v2，ADR-008）
# v2 变更：不再检查外部 SKILL_ROOT/AKG checkout——资产已 vendor 进仓，本脚本纯内部校验。
# 用法：bash deps/setup.sh --verify
set -euo pipefail
MODE="${1:--verify}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0

echo "[setup] mode=$MODE  root=$ROOT"

echo "[setup] vendored skill 组:"
for g in core triton-ascend ascendc pypto tilelang akg; do
  n=$(find "$ROOT/knowledge/skills/$g" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  echo "  $g: $n"
done

echo "[setup] third_party:"
if [ -f "$ROOT/third_party/akg/PINNED_COMMIT" ]; then
  echo "  ok      akg pinned: $(cat "$ROOT/third_party/akg/PINNED_COMMIT")"
else
  echo "  FAIL    third_party/akg/PINNED_COMMIT 缺失"; FAIL=1
fi

echo "[setup] python:"
python -c "import sys; assert sys.version_info >= (3,10); print('  ok      python', sys.version.split()[0])" || FAIL=1
python -c "import yaml" 2>/dev/null && echo "  ok      PyYAML" || echo "  warn    PyYAML 缺失（query.py 走回退解析）"

echo "[setup] 推荐：python tools/check_env.py 做全量自检（manifest hash/索引断链/NPU/模型端点）"

exit $FAIL
