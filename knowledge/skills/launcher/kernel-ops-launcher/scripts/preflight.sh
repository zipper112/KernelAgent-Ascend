#!/bin/bash
# preflight.sh —— 阶段①控制机环境自检（只读；输出 JSON 判定，root agent 消费）
# 用法：bash preflight.sh [repo_root]   （默认脚本位置上溯 5 级 = 仓根）
set -uo pipefail
REPO="${1:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

j() { printf '%s\n' "$1"; }

pass=""; fail=""; warn=""
chk() {  # chk <name> <ok|warn|fail> <detail>
  case "$2" in
    ok) pass="$pass $1," ;;
    warn) warn="$warn {\"$1\": \"${3//\"/\\\"}\"}," ;;
    fail) fail="$fail {\"$1\": \"${3//\"/\\\"}\"}," ;;
  esac
}

# --- python ---
PY=""
for c in python3 python; do
  if command -v $c >/dev/null 2>&1; then
    v=$($c -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0)
    if $c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY=$c; chk python ok "$v"; break
    fi
  fi
done
[ -z "$PY" ] && chk python fail "python>=3.10 not found (conda/venv?)"

# --- pyyaml ---
if [ -n "$PY" ] && $PY -c "import yaml" 2>/dev/null; then chk pyyaml ok; else chk pyyaml fail "pip install --user pyyaml"; fi

# --- git ---
command -v git >/dev/null 2>&1 && chk git ok || chk git fail "git missing"

# --- codex ---
CODEX_BIN=""
if command -v codex >/dev/null 2>&1; then
  CODEX_BIN=codex
elif [ -x "$HOME/.local/codex-agent/node_modules/.bin/codex" ]; then
  CODEX_BIN="$HOME/.local/codex-agent/node_modules/.bin/codex"
fi
if [ -n "$CODEX_BIN" ]; then
  CV=$("$CODEX_BIN" --version 2>/dev/null | head -1)
  # 0.157+ 只支持 responses 协议（chat 已移除）——版本可用性由冒烟最终判定
  chk codex ok "$CV"
else
  chk codex fail "codex CLI not found; options: npm i --prefix ~/.local/codex-agent @openai/codex (non-global)"
fi

# --- ~/.codex/config.toml GLM 接线 ---
CFG="$HOME/.codex/config.toml"
if [ -f "$CFG" ]; then
  if grep -q 'wire_api = "responses"' "$CFG" && grep -q "open.bigmodel.cn/api/v1" "$CFG"; then
    chk codex-glm-config ok
  else
    chk codex-glm-config warn "config.toml 存在但无 GLM responses 接线（需 base_url=/api/v1 + wire_api=responses）"
  fi
else
  chk codex-glm-config fail "~/.codex/config.toml 不存在（需生成 GLM 配置草稿并征得用户同意）"
fi

# --- GLM key ---
SEC="$REPO/agent-config/local-secrets.yaml"
if [ -f "$SEC" ] && grep -qE 'api_key:\s*"?[A-Za-z0-9._-]+' "$SEC"; then
  chk glm-key ok
else
  chk glm-key fail "agent-config/local-secrets.yaml 缺失或无 api_key（向用户索要，chmod 600，不回显）"
fi

# --- 冒烟（key+config 都在才做）---
if [ -n "$CODEX_BIN" ] && [ -f "$SEC" ]; then
  KEY=$($PY - "$SEC" << 'EOF' 2>/dev/null
import re, sys
for line in open(sys.argv[1], encoding="utf-8"):
    m = re.search(r'api_key:\s*"?([A-Za-z0-9._-]+)', line)
    if m:
        print(m.group(1)); break
EOF
)
  if [ -n "${KEY:-}" ]; then
    OUT=$(GLM_API_KEY="$KEY" timeout 180 "$CODEX_BIN" exec -s read-only \
      --skip-git-repo-check "reply with exactly: ok" 2>&1 | tail -3)
    if echo "$OUT" | grep -q "ok"; then
      chk codex-glm-smoke ok
    else
      chk codex-glm-smoke fail "smoke failed: $(echo "$OUT" | tail -1 | cut -c1-120)"
    fi
  fi
else
  chk codex-glm-smoke warn "跳过（key/config 未就绪）"
fi

# --- 输出 ---
fail="${fail%,}"; warn="${warn%,}"
j "{"
j "  \"repo\": \"$REPO\","
j "  \"pass\": [${pass%,}],"
j "  \"warn\": [${warn}],"
j "  \"fail\": [${fail}],"
j "  \"ready\": $([ -z "$fail" ] && echo true || echo false)"
j "}"
[ -z "$fail" ] && exit 0 || exit 1
