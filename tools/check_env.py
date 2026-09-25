#!/usr/bin/env python3
"""tools/check_env.py —— KDA-Ascend 环境自检。

用法：
  python tools/check_env.py           # 基础（开发机可跑；NPU 项报 pending）
  python tools/check_env.py --full    # 上板机全量（CANN/torch_npu/triton/msprof/akg）

输出分级：ok / warn / pending-NPU / fail；退出码 0=无 fail（pending 不算失败）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[dict] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail})
    mark = {"ok": "[ok] ", "warn": "[warn]", "pending-NPU": "[pend]", "fail": "[FAIL]"}[status]
    print(f"  {mark}  {name}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        record("python >= 3.10", "ok", f"{v.major}.{v.minor}.{v.micro} @ {sys.executable}")
    else:
        record("python >= 3.10", "fail", f"当前 {v.major}.{v.minor}")


def check_yaml() -> None:
    try:
        import yaml  # noqa: F401
        record("PyYAML", "ok")
    except ImportError:
        record("PyYAML", "warn", "query.py 将走回退解析；建议 pip install pyyaml")


def check_repo_layout() -> None:
    missing = [p for p in ("harness/core/README.md", "knowledge/router/index.yaml",
                           "knowledge/prompts/contract-template.md", "agent-config/models.yaml",
                           "docs/design/interaction-protocol.md", "deps/skills.yaml",
                           "deps/vendor-manifest.yaml", "THIRD_PARTY_NOTICES.md")
               if not (ROOT / p).exists()]
    record("仓库骨架完整性", "ok" if not missing else "fail", "缺: " + ",".join(missing) if missing else "")


def check_vendored_assets() -> None:
    """ADR-008 自包含校验：manifest 资产存在 + index ref 仓内可解析 + akg 钉版标记。"""
    import yaml
    try:
        data = yaml.safe_load((ROOT / "deps" / "vendor-manifest.yaml").read_text(encoding="utf-8"))
        assets = data.get("assets", [])
        missing = [a["name"] for a in assets if not (ROOT / a["path"]).exists()]
        record("vendored 资产（manifest）", "ok" if not missing else "fail",
               f"{len(assets)} 条资产" + (f"，断链: {missing[:3]}" if missing else ""))
    except Exception as e:  # noqa: BLE001
        record("vendored 资产（manifest）", "fail", str(e)[:100])
        return
    # index ref 解析（复用 query.py 的 loader）
    sys.path.insert(0, str(ROOT / "knowledge" / "router"))
    try:
        from query import load_index
        entries = load_index()
        broken = [e.get("id") for e in entries if not (ROOT / e.get("ref", "??")).exists()]
        record("知识索引 ref 仓内可解析", "ok" if not broken else "fail",
               f"{len(entries)} 条" + (f"，断链: {broken[:3]}" if broken else ""))
    except Exception as e:  # noqa: BLE001
        record("知识索引 ref 仓内可解析", "fail", str(e)[:100])
    pin = ROOT / "third_party" / "akg" / "PINNED_COMMIT"
    record("akg 钉版标记", "ok" if pin.exists() else "fail", pin.read_text().strip() if pin.exists() else "缺失")


def check_router() -> None:
    """路由可用性：跑三条代表性查询。"""
    cases = [
        (["--symptom", "mte2-bound", "--op-family", "norm", "--arch", "dav_2201"], 0),
        (["--entry", "norm-family-opt"], 0),
        (["--symptom", "conv-nonexistent"], 1),   # 无命中 → 盲区路径也必须正确
    ]
    q = ROOT / "knowledge" / "router" / "query.py"
    for args, expect in cases:
        r = subprocess.run([sys.executable, str(q), *args, "--compact"],
                           capture_output=True, text=True, timeout=30)
        record(f"router query {' '.join(args)}", "ok" if r.returncode == expect else "fail",
               f"exit={r.returncode}" + (f" stderr={r.stderr.strip()[:80]}" if r.returncode != expect else ""))


def check_skills() -> None:
    """v2：vendored 自包含校验（替代旧的外部 SKILL_ROOT 检查，ADR-008）。"""
    base = ROOT / "knowledge" / "skills"
    groups = {g.name: sum(1 for d in g.iterdir() if d.is_dir()) for g in base.iterdir() if g.is_dir()}
    total = sum(groups.values())
    expect_min = {"core": 13, "triton-ascend": 6, "ascendc": 24, "pypto": 17, "tilelang": 6}
    short = {k: v for k, v in expect_min.items() if groups.get(k, 0) < v}
    detail = " ".join(f"{k}={v}" for k, v in sorted(groups.items()))
    record(f"vendored skill 资产（{total} 目录）", "ok" if not short else "fail",
           detail + (f"；缺: {short}" if short else ""))


def check_upstreams() -> None:
    """v2：上游 checkout 检查改为可选（仅 sync_assets 用；运行链不依赖）。"""
    sync = ROOT / "tools" / "sync_assets.py"
    record("资产同步工具（可选）", "ok" if sync.exists() else "warn",
           "上游 checkout 缺失不影响运行（ADR-008 自包含）" if sync.exists() else "缺 tools/sync_assets.py")


def check_npu(full: bool) -> None:
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        record("npu-smi", "pending-NPU" if not full else "fail", "本机无 NPU 工具（上板机重跑 --full）")
        return
    record("npu-smi", "ok", npu_smi)
    for mod, label in (("torch_npu", "torch_npu"), ("triton", "triton(triton-ascend 待验)")):
        r = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"],
                           capture_output=True, text=True, timeout=60)
        record(label, "ok" if r.returncode == 0 else ("fail" if full else "pending-NPU"),
               r.stdout.strip() or r.stderr.strip()[:80])
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("CANN_HOME")
    record("CANN 环境变量", "ok" if cann else ("fail" if full else "pending-NPU"), str(cann))
    msprof = shutil.which("msprof")
    record("msprof", "ok" if msprof else ("fail" if full else "pending-NPU"), str(msprof))


def check_model_endpoint() -> None:
    """模型端点连通：经 infra/secrets 取 key（优先环境变量），实测一次 flash 调用（≤4 tokens 成本）。"""
    sys.path.insert(0, str(ROOT))
    from infra.secrets.provider import get_glm_key
    key = get_glm_key()
    if not key:
        record("GLM 端点连通", "warn", "无 key（GLM_API_KEY 或 local-secrets.yaml）")
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
            data=json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "ping"}],
                             "max_tokens": 4}).encode(),
            headers={"Authorization": f"Bearer {key.get()}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
            record("GLM 端点连通", "ok", f"HTTP {resp.status}, usage={body.get('usage', {}).get('total_tokens')} tokens")
    except Exception as e:  # noqa: BLE001
        record("GLM 端点连通", "fail", str(e)[:120])


def check_remote(host_arg: str | None) -> None:
    """远程 NPU 探测（infra/remote；--remote [jump:host] 时替代本机 NPU 项）。"""
    if not host_arg:
        return
    sys.path.insert(0, str(ROOT))
    from infra.remote.executor import RemoteExecutor, RemoteTarget
    jump, _, host = host_arg.partition(":")
    ex = RemoteExecutor(RemoteTarget(jump or "jump", host or "yq-e15"))
    try:
        r = ex.probe()
    except Exception as e:  # noqa: BLE001
        record(f"remote {host_arg}", "fail", str(e)[:120])
        return
    if not r["reachable"]:
        record(f"remote {host_arg}", "fail", "不可达（两跳 SSH 检查 ~/.ssh/config）")
        return
    c = r["checks"]
    record(f"remote {host_arg} 可达", "ok")
    record("  remote NPU", "ok" if c.get("npu", "none") != "none" else "fail", str(c.get("npu")))
    record("  remote CANN toolkit", "ok" if c.get("cann_toolkit") == "ok" else "warn",
           "missing → Phase 0 上板装（见 maintenance.md）" if c.get("cann_toolkit") == "missing" else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="上板机全量（NPU 项 fail 计入）")
    ap.add_argument("--remote", nargs="?", const="jump:yq-e15", metavar="[jump:host]",
                    help="远程 NPU 探测（infra/remote；默认 jump:yq-e15）")
    ap.add_argument("--json", action="store_true", help="额外输出 JSON 到 stdout 尾部")
    args = ap.parse_args()

    print(f"KDA-Ascend check_env  platform={platform.system()}  full={args.full}  remote={args.remote}")
    print("[1/7] 基础");      check_python(); check_yaml()
    print("[2/7] 仓库");      check_repo_layout()
    print("[3/7] 知识路由");  check_router(); check_vendored_assets()
    print("[4/7] skill 资产"); check_skills()
    print("[5/7] 资产同步");  check_upstreams()
    print("[6/7] NPU/模型");  check_npu(args.full); check_model_endpoint()
    print("[7/7] 远程");      check_remote(args.remote)

    fails = sum(1 for r in RESULTS if r["status"] == "fail")
    warns = sum(1 for r in RESULTS if r["status"] == "warn")
    pends = sum(1 for r in RESULTS if r["status"] == "pending-NPU")
    print(f"\n汇总: {len(RESULTS)} 项 — fail={fails} warn={warns} pending={pends}")
    if args.json:
        print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
