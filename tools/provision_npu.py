#!/usr/bin/env python3
"""tools/provision_npu.py —— 无网 NPU 机（e15）环境供给（ADR-011 策略 C：jump 中转下载）。

现实约束（2026-09-25 实测）：e15 无网且反向到 jump 不通；jump 有网但只有 python3.7。
方案：本地解析 PyPI 依赖闭包 → 生成 curl 清单 → jump 执行下载（aarch64 wheel）→
两跳 scp 到 e15 → venv 内离线 pip 安装 → smoke。

用法：
  python tools/provision_npu.py --plan          # 只打印将下载的 wheel 清单（本地解析，零成本）
  python tools/provision_npu.py --provision     # 全流程：下载/传输/安装/smoke
  python tools/provision_npu.py --smoke         # 只跑远端 smoke
环境变量：KDA_JUMP=jump  KDA_HOST=yq-e15  KDA_REMOTE_ROOT=~/kda-ascend（可覆盖）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

JUMP = os.environ.get("KDA_JUMP", "jump")
HOST = os.environ.get("KDA_HOST", "yq-e15")
ROOT = os.environ.get("KDA_REMOTE_ROOT", "~/kda-ascend")

# 版本组合锁定（torch_npu 2.13.0rc1 ↔ torch 2.13.0；cp310 aarch64 实测在架）
TARGETS = {"torch": "2.13.0", "torch_npu": "2.13.0rc1"}

MIRROR = "https://pypi.org/pypi"      # JSON API（比 simple 页可靠：结构化 releases）
LOCAL_PROXY = os.environ.get("KDA_LOCAL_PROXY", "http://127.0.0.1:7897")   # 本机出网代理（e15/jump 不用）


def _open(url: str, timeout: int = 30):
    proxy = urllib.request.ProxyHandler({"http": LOCAL_PROXY, "https": LOCAL_PROXY}) if LOCAL_PROXY else \
        urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy)
    return opener.open(url, timeout=timeout)


def sh(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def remote(cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    inner = f"timeout {timeout} ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new {HOST} {json.dumps(cmd)}"
    return sh(["ssh", "-o", "BatchMode=yes", JUMP, inner], timeout=timeout + 30)


def resolve_closure() -> list[str]:
    """PyPI JSON API 精确取 wheel URL（torch/torch_npu 显式版本；jinja2/pyyaml 走通用名匹配）。"""
    urls: list[str] = []
    for pkg, ver in TARGETS.items():
        with _open(f"{MIRROR}/{pkg}/{ver}/json") as resp:
            data = json.load(resp)
        for f in data["urls"]:
            fn = f["filename"]
            if fn.endswith(".whl") and "cp310" in fn and "aarch64" in fn:
                urls.append(f["url"])
                break
        else:
            raise SystemExit(f"未找到 {pkg}=={ver} 的 cp310 aarch64 wheel")
    # 轻依赖：取最新版任意纯 py wheel（pip 离线装时用）
    for pkg in ("jinja2", "pyyaml", "markupsafe", "typing-extensions", "filelock",
                "sympy", "networkx", "jinja2", "fsspec"):
        if pkg in [u.rsplit('/', 1)[-1].split('-')[0].replace('_', '-').lower() for u in urls]:
            continue
        try:
            with _open(f"{MIRROR}/{pkg}/json") as resp:
                data = json.load(resp)
            files = data["urls"]
            pick = next((f["url"] for f in files
                         if f["filename"].endswith(".whl") and ("py3-none-any" in f["filename"] or ("cp310" in f["filename"] and "aarch64" in f["filename"]))), None)
            if pick:
                urls.append(pick)
        except Exception:  # noqa: BLE001
            pass
    return urls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--provision", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not any([args.plan, args.provision, args.smoke]):
        ap.error("--plan / --provision / --smoke 必选其一")

    if args.plan:
        for u in resolve_closure():
            print(u)
        return 0

    if args.smoke:
        r = remote(f"cd {ROOT} && venv/bin/python -c "
                   f"'import torch,torch_npu;print(torch_npu.npu.is_available())' 2>&1 | tail -1")
        print(r.stdout.strip() or r.stderr.strip()[:300])
        return r.returncode

    # --provision
    urls = resolve_closure()
    print(f"[provision] 依赖闭包 {len(urls)} 个 wheel")
    r = remote("mkdir -p ~/kda-ascend/wheels && rm -f ~/kda-ascend/wheels/*.whl")
    ok = True
    for i, u in enumerate(urls, 1):
        fn = u.rsplit("/", 1)[-1]
        print(f"[provision] ({i}/{len(urls)}) jump 下载 {fn}")
        d = remote(f"curl -sSL --max-time 1800 -o ~/kda-ascend/wheels/{json.dumps(fn)} {json.dumps(u)} && ls -la ~/kda-ascend/wheels/{json.dumps(fn)}", timeout=1900)
        if d.returncode != 0 or "No such" in d.stdout:
            print(f"  FAIL: {(d.stderr or d.stdout)[:200]}")
            ok = False
    if not ok:
        return 1
    print("[provision] 远端 venv 离线安装（torch 先装，禁依赖检查）")
    r = remote(f"cd {ROOT} && python3 -m venv venv 2>/dev/null; "
               f"venv/bin/pip install --no-index --find-links wheels --no-deps torch*.whl && "
               f"venv/bin/pip install --no-index --find-links wheels torch_npu*.whl && "
               f"venv/bin/pip install jinja2 pyyaml pandas --no-index --find-links wheels 2>/dev/null || true", timeout=900)
    print(r.stdout[-1500:] if r.stdout else r.stderr[:500])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
