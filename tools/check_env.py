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
                           "docs/design/interaction-protocol.md", "deps/skills.yaml")
               if not (ROOT / p).exists()]
    record("仓库骨架完整性", "ok" if not missing else "fail", "缺: " + ",".join(missing) if missing else "")


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
    skill_root = Path(os.environ.get("SKILL_ROOT", Path.home() / ".agents" / "skills"))
    names = ["ops-profiling", "ascendc-performance-best-practices", "ascendc-perf-optimize",
             "ascendc-crash-debug", "ascendc-precision-debug", "ascendc-runtime-debug",
             "ascendc-env-check", "triton-op-coding", "triton-op-designer", "pypto-golden-generate"]
    missing = [n for n in names if not (skill_root / n / "SKILL.md").exists()]
    if not missing:
        record("核心 skill 依赖（10 项）", "ok", f"@ {skill_root}")
    else:
        record("核心 skill 依赖（10 项）", "warn", "缺: " + ",".join(missing) + f"（skill_root={skill_root}）")


def check_upstreams() -> None:
    for name, sub in (("akg", "akg_agents"), ("humanize", None),
                      ("kda", None), ("mlsys2026-flashinfer-contest", None)):
        p = ROOT.parent / ".repo-research" / name
        ok = p.exists() and ((p / sub).exists() if sub else True)
        record(f"上游 checkout: {name}", "ok" if ok else "warn", str(p))


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
    """模型端点连通（有 GLM_API_KEY 才测；无则 pending）。"""
    key = os.environ.get("GLM_API_KEY")
    if not key:
        record("GLM 端点连通", "pending-NPU", "未设 GLM_API_KEY（上板/运行期再验）")
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            data=json.dumps({"model": "glm-4.x-standard", "messages": [{"role": "user", "content": "ping"}],
                             "max_tokens": 4}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20):
            record("GLM 端点连通", "ok", "chat/completions 200")
    except Exception as e:  # noqa: BLE001
        record("GLM 端点连通", "warn", str(e)[:120])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="上板机全量（NPU 项 fail 计入）")
    ap.add_argument("--json", action="store_true", help="额外输出 JSON 到 stdout 尾部")
    args = ap.parse_args()

    print(f"KDA-Ascend check_env  platform={platform.system()}  full={args.full}")
    print("[1/6] 基础");      check_python(); check_yaml()
    print("[2/6] 仓库");      check_repo_layout()
    print("[3/6] 知识路由");  check_router()
    print("[4/6] skill 依赖"); check_skills()
    print("[5/6] 上游");      check_upstreams()
    print("[6/6] NPU/模型");  check_npu(args.full); check_model_endpoint()

    fails = sum(1 for r in RESULTS if r["status"] == "fail")
    warns = sum(1 for r in RESULTS if r["status"] == "warn")
    pends = sum(1 for r in RESULTS if r["status"] == "pending-NPU")
    print(f"\n汇总: {len(RESULTS)} 项 — fail={fails} warn={warns} pending={pends}")
    if args.json:
        print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
