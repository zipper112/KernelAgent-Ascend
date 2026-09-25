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
import os
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


CANN_OPS_REPOS = {
    # 三大仓：vendor 时取算子目录（KEEP），剥 tests/docs/cmake/scripts/examples/torch_extension/build 基建
    "ops-transformer": ["attention", "ffn", "gmm", "mamba", "mc2", "mhc", "moe", "posembedding",
                        "common", "experimental"],
    "ops-nn": ["activation", "common", "control", "conv", "experimental", "foreach", "hash",
               "index", "loss", "matmul", "norm", "optim", "pooling", "quant", "rnn", "vfusion"],
    "ops-math": ["conversion", "math", "random", "common", "experimental"],
}
CANN_OPS_SMALL = ["ops-blas", "ops-collections", "ops-cv", "ops-fft", "ops-sparse", "ops-gnn"]
CANN_OPS_META = ["LICENSE", "README.md", "CHANGELOG.md", "classify_rule.yaml", "version.cmake",
                 "version.info", "Third_Party_Open_Source_Software_List.yaml",
                 "Third_Party_Open_Source_Software_Notice"]


def bootstrap_cann_ops(proxy: str | None) -> int:
    """重资产重建（ADR-009 修订版）：拉取十仓 → 剥 tests → 组装 third_party/cann-ops → 重建 production-index。

    本地 third_party/cann-ops/ 不入 git（.gitignore）；迁移新机器后跑本命令即恢复生产代码层。
    """
    import shutil
    import subprocess as sp

    dest_root = ROOT / "third_party" / "cann-ops"
    if dest_root.exists():
        print(f"已存在 {dest_root}（如需重建先手动删除）")
        return 1
    env = dict(os.environ)
    if proxy:
        env["https_proxy"] = env["http_proxy"] = proxy
    tmp = ROOT / "third_party" / ".cann-ops-tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    def run(cmd: list[str]) -> None:
        r = sp.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"命令失败 {cmd[0]}: {r.stderr[:300]}")

    for repo, families in CANN_OPS_REPOS.items():
        print(f"[bootstrap] clone {repo} ...")
        run(["git", "clone", "--depth", "1", f"https://gitcode.com/cann/{repo}.git", str(tmp / repo)])
        dest = dest_root / repo
        dest.mkdir(parents=True)
        for fam in families:
            src = tmp / repo / fam
            if src.exists():
                shutil.copytree(src, dest / fam)
        for meta in CANN_OPS_META:
            src = tmp / repo / meta
            if src.exists():
                shutil.copy2(src, dest / meta)
    small = dest_root / "small"
    small.mkdir(parents=True)
    for repo in CANN_OPS_SMALL:
        print(f"[bootstrap] clone {repo} ...")
        run(["git", "clone", "--depth", "1", f"https://gitcode.com/cann/{repo}.git", str(tmp / repo)])
        shutil.copytree(tmp / repo, small / repo, ignore=shutil.ignore_patterns(".git"))
    # 剥 tests（体积大头）
    n = 0
    for d in dest_root.rglob("*"):
        if d.is_dir() and d.name in ("tests", "test", "ut", "st"):
            shutil.rmtree(d)
            n += 1
    print(f"[bootstrap] 剥除 {n} 个 tests 目录")
    shutil.rmtree(tmp)
    # 重建索引
    run([sys.executable, str(ROOT / "tools" / "build_production_index.py")])
    print(f"[bootstrap] 完成：{dest_root}（重建索引见上）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="对比外部源与仓内 hash，输出 diff 报告")
    ap.add_argument("--sync", metavar="NAME", help="重拉指定资产（默认仅报告，--apply 才落盘）")
    ap.add_argument("--apply", action="store_true", help="配合 --sync 真正写入")
    ap.add_argument("--bootstrap-cann-ops", action="store_true",
                    help="重建重资产 third_party/cann-ops（拉十仓+剥tests+建索引；迁移新机器后跑）")
    ap.add_argument("--proxy", default=None, help="代理地址（如 http://127.0.0.1:7897）")
    args = ap.parse_args()

    if args.bootstrap_cann_ops:
        return bootstrap_cann_ops(args.proxy)

    assets = parse_manifest()
    if not any([args.check, args.sync]):
        ap.error("--check、--sync 或 --bootstrap-cann-ops 必选其一")

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
