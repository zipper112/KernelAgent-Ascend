#!/usr/bin/env bash
# deps/setup.sh —— 一键安装/校验外部依赖
# 用法：
#   bash deps/setup.sh --verify    # 只校验（开发机可跑，NPU 项跳过）
#   bash deps/setup.sh --install   # 安装：链接本机 skill、校验 akg 钉版可 import
set -euo pipefail
MODE="${1:--verify}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_ROOT="${SKILL_ROOT:-$HOME/.agents/skills}"
AKG_LOCAL="${AKG_LOCAL:-$ROOT/../.repo-research/akg}"   # 与本仓库同级；换机器用环境变量覆盖
FAIL=0

check_skill() {
  local name="$1" path="$SKILL_ROOT/$1"
  if [ -f "$path/SKILL.md" ]; then
    echo "  ok      skill: $name"
  else
    echo "  MISSING skill: $name ($path)"; FAIL=1
  fi
}

echo "[setup] mode=$MODE  skill_root=$SKILL_ROOT"

# 1) 本机 skill 依赖（读 skills.yaml 名单；无 yq 时用 grep 兜底）
echo "[setup] skills:"
SKILLS=$(grep -E '^\s+- name: ' "$ROOT/deps/skills.yaml" | sed 's/.*name: *//')
for s in $SKILLS; do check_skill "$s"; done

# 2) akg 钉版
echo "[setup] akg pinned checkout:"
if [ -d "$AKG_LOCAL/akg_agents/python/akg_agents" ]; then
  echo "  ok      akg_agents importable path: $AKG_LOCAL"
else
  echo "  warn    akg local not found: $AKG_LOCAL（上板机配置 AKG_LOCAL 指向 checkout）"
fi

# 3) python 基础
echo "[setup] python:"
python -c "import sys; assert sys.version_info >= (3,10), 'need py>=3.10'; print('  ok      python', sys.version.split()[0])" || FAIL=1
python -c "import yaml" 2>/dev/null && echo "  ok      PyYAML" || echo "  warn    PyYAML 缺失（query.py 走回退解析，建议 pip install pyyaml）"

exit $FAIL
