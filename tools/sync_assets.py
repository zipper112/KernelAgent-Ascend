#!/usr/bin/env python3
"""tools/sync_assets.py —— vendored 资产的可选更新工具。

本工具不是运行依赖（check_env/pytest 全部只看仓内）；仅在想从外部源拉新版时使用：
  python tools/sync_assets.py --check    # 只对比：外部源 vs 仓内 manifest 的 hash diff 报告
  python tools/sync_assets.py --sync <name> [--source local|akg]   # 重拉单个资产（需人工确认后手动更新 manifest）

hash 漂移的处理纪律（maintenance.md）：先 --check 看 diff → 评估影响（被 index.yaml 引用的条目
必须重跑路由验证）→ 同步 → 更新 vendor-manifest.yaml → commit 前缀 deps:。
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "deps" / "vendor-manifest.yaml"
LOCAL_SKILL_ROOT = Path(r"C:\Users\11565\.agents\skills")
AKG_SKILLS = ROOT.parent / ".repo-research" / "akg" / "akg_agents" / "python" / "akg_agents" / "op" / "resources" / "skills"
AKG_CODE = ROOT.parent / ".repo-research" / "akg" / "akg_agents" / "python" / "akg_agents"


def tree_sha(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(p)).replace("\\", "/").encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def parse_manifest() -> list[dict]:
    import yaml
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return data.get("assets", [])


def source_path(name: str, group: str) -> Path | None:
    if group == "akg":
        return AKG_SKILLS / name
    if group == "third_party":
        return AKG_CODE / "op" if name == "akg-verifier-code" else None
    return LOCAL_SKILL_ROOT / name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="对比外部源与仓内 hash，输出 diff 报告")
    ap.add_argument("--sync", metavar="NAME", help="重拉指定资产（默认仅报告，--apply 才落盘）")
    ap.add_argument("--apply", action="store_true", help="配合 --sync 真正写入")
    args = ap.parse_args()

    assets = parse_manifest()
    if not any([args.check, args.sync]):
        ap.error("--check 或 --sync 必选其一")

    if args.check:
        diffs, missing_src, ok = [], [], 0
        for a in assets:
            src = source_path(a["name"], a.get("group", ""))
            if not src or not src.exists():
                missing_src.append(a["name"])
                continue
            new_sha = tree_sha(src)
            if new_sha != a["tree_sha256"]:
                diffs.append((a["name"], a["tree_sha256"], new_sha))
            else:
                ok += 1
        print(f"vendored 资产对比：{ok} 一致 / {len(diffs)} 漂移 / {len(missing_src)} 外部源不可达（本机无源不影响仓内使用）")
        for name, old, new in diffs:
            print(f"  DRIFT {name}: {old} -> {new}")
        for name in missing_src:
            print(f"  NOSRC {name}")
        return 0

    if args.sync:
        target = next((a for a in assets if a["name"] == args.sync), None)
        if not target:
            print(f"manifest 中无此资产: {args.sync}")
            return 2
        src = source_path(target["name"], target.get("group", ""))
        if not src or not src.exists():
            print(f"外部源不可达: {src}")
            return 2
        dst = ROOT / target["path"]
        if args.apply:
            subprocess.run(["robocopy", str(src), str(dst), "/MIR", "/XD", "__pycache__", ".git"] + ([] if True else []), shell=True)
            # robocopy 返回码 0-7 为成功
            new_sha = tree_sha(dst)
            print(f"已同步 {target['name']}，新 hash {new_sha}——请更新 vendor-manifest.yaml 并重跑路由验证")
        else:
            print(f"[dry-run] 将 {src} -> {dst}；加 --apply 落盘")
        return 0


if __name__ == "__main__":
    sys.exit(main())
