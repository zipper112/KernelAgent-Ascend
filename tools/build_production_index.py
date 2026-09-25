#!/usr/bin/env python3
"""tools/build_production_index.py —— 生产算子代码索引生成器（KernelWiki 式）。

扫描 third_party/cann-ops/ 下的官方算子仓，产出 knowledge/router/production-index.yaml：
每条 = {op, repo, path, archs, dsl, kinds, note}。
架构识别：路径含 archXX / ascend910b / ascend950 / davXXX 等标记自动归档。
重新生成：python tools/build_production_index.py（vendor 更新后必跑）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANN_OPS = ROOT / "third_party" / "cann-ops"
OUT = ROOT / "knowledge" / "router" / "production-index.yaml"

ARCH_PATTERNS = [
    (r"arch200|aiv200|aicore200|ascend310\b", "dav_200"),
    (r"arch22|aiv22|ascend910b|ascend910_93|dav2201|a2|a3(?![0-9])", "dav_2201"),
    (r"arch35|aiv35|ascend950|dav3510|vf_|regbase|blaze", "dav_3510"),
]

# 算子名提取：路径形如 <repo>/<family>/<op_name>/... 或 <repo>/<op_name>/...
FAMILY_HINT = {
    "ops-transformer": ["attention", "ffn", "gmm", "mamba", "mc2", "mhc", "moe", "posembedding"],
    "ops-nn": ["activation", "control", "conv", "foreach", "hash", "index", "loss", "matmul",
               "norm", "optim", "pooling", "quant", "rnn", "vfusion"],
    "ops-math": ["conversion", "math", "random"],
    "small": ["ops-blas", "ops-collections", "ops-cv", "ops-fft", "ops-sparse", "ops-gnn"],
}


def detect_archs(rel: str) -> list[str]:
    hits = set()
    for pat, arch in ARCH_PATTERNS:
        if re.search(pat, rel, re.IGNORECASE):
            hits.add(arch)
    return sorted(hits)


def main() -> int:
    if not CANN_OPS.exists():
        print(f"未找到 {CANN_OPS}——先按 deps/vendor-manifest 的来源 vendor", file=sys.stderr)
        return 2
    entries: list[dict] = []
    for repo in sorted(CANN_OPS.iterdir()):
        if not repo.is_dir():
            continue
        repo_name = repo.name                      # ops-transformer / ops-nn / ops-math / small
        if repo_name == "small":
            for sub in sorted(repo.iterdir()):
                if sub.is_dir():
                    entries += scan_repo(sub, sub.name)          # 以 small/ops-blas 为 repo
        else:
            entries += scan_repo(repo, repo_name)
    # 去重合并（op+path 唯一）
    seen: dict[str, dict] = {}
    for e in entries:
        key = f"{e['repo']}:{e['path']}"
        if key in seen:
            seen[key]["archs"] = sorted(set(seen[key]["archs"]) | set(e["archs"]))
        else:
            seen[key] = e
    entries = sorted(seen.values(), key=lambda e: (e["repo"], e["op"]))

    lines = ["# knowledge/router/production-index.yaml —— 生产算子代码索引（KernelWiki 式）",
             "# 生成：python tools/build_production_index.py（vendor 更新后重跑）",
             "# 用途：知识检索的生产实现层——症状/算子族命中后，agent 到这里拿真实生产源码路径。",
             "# 排除：tests 目录已剥（vendor 时）；experimental 目录保留（前沿参考）。",
             "version: 1",
             f"generated_from: third_party/cann-ops",
             f"entries:"]
    for e in entries:
        archs = "[" + ", ".join(e["archs"]) + "]" if e["archs"] else "[both]"
        lines.append(f"  - {{op: {e['op']}, repo: {e['repo']}, path: {e['path']}, archs: {archs}, kind: {e['kind']}}}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"production-index.yaml: {len(entries)} 条目 -> {OUT.relative_to(ROOT)}")
    return 0


def scan_repo(repo_dir: Path, repo_name: str) -> list[dict]:
    """扫一个仓的算子目录（含 op_kernel/op_host 的目录视为一个算子单元）。"""
    out: list[dict] = []
    top_families = FAMILY_HINT.get(repo_name, [])
    roots: list[Path] = []
    for fam in top_families:
        fam_dir = repo_dir / fam
        if fam_dir.is_dir():
            roots.append(fam_dir)
    for r in roots:
        for op_dir in sorted(r.iterdir()):
            if not op_dir.is_dir() or op_dir.name in ("common", "experimental", "docs"):
                continue
            if (op_dir / "op_kernel").exists() or (op_dir / "op_host").exists():
                rel = op_dir.relative_to(ROOT).as_posix()
                out.append({"op": op_dir.name, "repo": repo_name, "path": rel,
                            "archs": detect_archs(rel), "kind": "op"})
    # 实验目录再深一层
    for r in roots:
        exp = r / "experimental"
        if exp.is_dir():
            for op_dir in sorted(exp.iterdir()):
                if op_dir.is_dir() and ((op_dir / "op_kernel").exists() or (op_dir / "op_host").exists()):
                    rel = op_dir.relative_to(ROOT).as_posix()
                    out.append({"op": op_dir.name, "repo": repo_name, "path": rel,
                                "archs": detect_archs(rel), "kind": "experimental-op"})
    return out


if __name__ == "__main__":
    sys.exit(main())
