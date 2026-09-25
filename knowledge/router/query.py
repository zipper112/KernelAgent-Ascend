#!/usr/bin/env python3
"""knowledge/router/query.py —— 三轴知识检索（症状 × 算子族 × 硬件代际）。

对标 KernelWiki query.py 的过滤维度与打分规则（title > tag > body），
数据源是 index.yaml。输出供 ctx 组装器 L1 注入与陪伴模式 agent 自查。

用法：
  python query.py --symptom mte2-bound --op-family norm --arch dav_2201
  python query.py --entry norm-family-opt            # 按 id 精确取
  python query.py --list-fundamentals --arch dav_2201
退出码：0 有命中；1 无命中（调用方应触发盲区条款）；2 参数/数据错误。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INDEX_PATH = Path(__file__).parent / "index.yaml"
PROD_INDEX_PATH = Path(__file__).parent / "production-index.yaml"   # 生产代码层（KernelWiki 式）

# 别名表：常见口语词 → 索引规范词（KernelWiki 别名扩展的简化版）
ALIASES = {
    "mem-bound": "memory-bound",
    "memory-bound": "memory-bound",
    "mte2": "mte2-bound",
    "vec": "vec-bound",
    "cube": "cube-bound",
    "scalar": "scalar-bound",
    "imbalance": "load-imbalance",
    "tail": "tail-effect",
    "bubble": "pipeline-bubble",
    "l2": "low-l2-hit",
    "nan": "tolerance-fail",
    "hang": "hang",
    "aic-error": "aic-error",
}

FUNDAMENTAL_KIND = "fundamental"


def load_index() -> list[dict]:
    """读 index.yaml 的 entries 列表（PyYAML 优先；无依赖时行级回退）。"""
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8"))
        return [dict(e, ref=e.get("path", e.get("ref", ""))) for e in data.get("entries", [])]
    except ImportError:
        pass
    # 无 PyYAML 时的行级回退解析（v0.1 修复：剥行内注释，与 yaml 路径行为一致）
    entries: list[dict] = []
    cur: dict | None = None
    kv = re.compile(r"^\s*-?\s*([\w][\w-]*):\s*(.+?)\s*$")
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*-\s+id:", line):
            if cur:
                entries.append(cur)
            cur = {}
        if cur is None:
            continue
        m = kv.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        raw = raw.split(" #")[0].strip()          # 剥行内注释（yaml 语义）
        if raw.startswith("[") and raw.endswith("]"):
            val = [v.strip().strip('"').strip("'") for v in raw[1:-1].split(",") if v.strip()]
        else:
            val = raw.strip('"').strip("'")
        cur[key] = val
        if key == "path":                 # v2 字段：path 即检索输出的 ref
            cur["ref"] = val
    if cur:
        entries.append(cur)
    return entries


def expand_symptoms(symptoms: list[str]) -> list[str]:
    """口语词 → 规范词归一（v0.1 修复：同义词合并，原词不再重复保留——
    重复词会稀释打分且让命中判定把没归一的词当独立症状）。"""
    out = []
    for s in symptoms:
        canon = ALIASES.get(s, s)
        if canon not in out:
            out.append(canon)
    return out


def score_entry(entry: dict, args: argparse.Namespace) -> int | None:
    """返回匹配分；None = 不匹配。title(=id) 10 分 > tag(symptom/op_family) 5 分 > 其他 1 分。"""
    return score_entry_impl(entry, args)


def score_entry_impl(entry: dict, args: argparse.Namespace) -> int | None:
    arch = entry.get("arch", "both")
    if args.arch and arch not in ("both", args.arch):
        return None
    if args.entry and entry.get("id") != args.entry:
        return None
    if args.kind and entry.get("kind", "").split("+")[0] != args.kind:
        return None
    if args.list_fundamentals and entry.get("kind") != FUNDAMENTAL_KIND:
        return None

    score = 0
    matched_sym = False
    wildcard_only = False
    if args.symptom:
        syms = expand_symptoms(args.symptom)
        esyms = entry.get("symptom", [])
        if isinstance(esyms, str):
            esyms = [esyms]
        if set(syms) & set(esyms):
            score += 5
            matched_sym = True
        elif "*" in esyms:
            score += 1          # 通配条目只在无专命中时兜底，且不记 matched
            wildcard_only = True
        elif not esyms:         # fundamental 无症状轴：仅当按 id 精确取时放行
            if not args.entry:
                return None
        else:
            return None
    if args.op_family:
        fams = entry.get("op_family", [])
        if isinstance(fams, str):
            fams = [fams]
        fam_hit = args.op_family in fams or "*" in fams
        if not fam_hit:
            if args.symptom and not matched_sym:
                return None
        else:
            score += 5 if args.op_family in fams else 3   # 精确族 > 通配族
    if args.entry:
        score += 10
    if args.symptom and wildcard_only and not args.entry:
        # 症状查询只被通配条目兜住：不算知识命中（调用方应触发盲区条款）
        return None
    if score == 0:
        score = 1 if (args.symptom or args.op_family) else 0
    return score


_VERSION_SUFFIX = re.compile(r"(?:_v|_)[0-9]+(?:_[0-9]+)*$")   # add_v2 / all_gather_matmul_v3 / a_1_2 尾部版本


def _op_tokens(op_name: str) -> set[str]:
    """生产代码层词元拆分（--production 精确匹配用）：rms_norm ⊃ "norm"；random_normal ⊉ "norm"（子串不算）。
    v0.1 修正（对齐真实 1210 条命名）：只剥**尾部版本段**（add_v2→add；accumulate_nv2 是历史名不是版本，
    保 nv2 词元），驼峰整体小写（RMSNorm→rmsnorm），空词元丢弃。"""
    base = _VERSION_SUFFIX.sub("", op_name)
    return {t.lower() for t in base.split("_") if t}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symptom", nargs="*", default=[], help="症状词（支持别名，见 ALIASES）")
    p.add_argument("--op-family", default=None)
    p.add_argument("--arch", default=None, choices=["dav_2201", "dav_3510"])
    p.add_argument("--kind", default=None, help="guide|case|template|threshold|fundamental")
    p.add_argument("--entry", default=None, help="按 id 精确取")
    p.add_argument("--list-fundamentals", action="store_true")
    p.add_argument("--production", action="store_true",
                   help="联合检索生产代码索引（cann-ops vendor 层）；--op-family 会自动追加")
    p.add_argument("--compact", action="store_true")
    args = p.parse_args()

    if not any([args.symptom, args.op_family, args.arch, args.entry, args.list_fundamentals, args.kind]):
        p.error("至少给一个过滤条件（或 --list-fundamentals）")

    try:
        entries = load_index()
    except FileNotFoundError:
        print(f"{{\"error\": \"index not found: {INDEX_PATH}\"}}")
        return 2

    # 生产代码层联合检索（仅 --production 显式开启）
    # 匹配规则：算子名按 _ 拆词元——词元完全等于 family（rms_norm ⊃ "norm"）才算核心命中；
    # 子串包含（random_normal 的 "normal" ≠ "norm"）不算，杜绝 abs/random 类误入。
    prod_entries: list[dict] = []
    if args.production and args.op_family and PROD_INDEX_PATH.exists():
        import yaml as _y
        pdata = _y.safe_load(PROD_INDEX_PATH.read_text(encoding="utf-8"))
        fam = args.op_family
        exact = []
        for e in pdata.get("entries", []):
            if args.arch and e.get("archs") != ["both"] and args.arch not in e.get("archs", []):
                continue
            tokens = _op_tokens(e["op"])
            if fam not in tokens:
                continue
            exact.append({**e, "id": f"prod:{e['op']}", "skill": f"cann-ops:{e['repo']}",
                          "ref": e["path"], "symptom": [], "kind": e.get("kind", "op"),
                          "op_family": [fam], "arch": ",".join(e.get("archs", [])),
                          "confidence": "verified"})
        prod_entries = exact[:15]

    scored = []
    for e in entries:
        s = score_entry_impl(e, args)
        if s is not None:
            scored.append((s, e))
    scored.sort(key=lambda t: -t[0])
    # 方法论条目在前；生产代码命中追加在后（词元精确匹配，无 fuzzy）
    final = scored + [(0, pe) for pe in prod_entries]

    if not final:
        print('{"hits": 0, "note": "NO MATCH -> trigger blindspot-declaration clause"}')
        return 1

    if args.compact:
        for _, e in final:
            print(f"[{e.get('kind','?')}] {e.get('id')}: {e.get('skill')} -> {e.get('ref')}")
    else:
        import json
        hits = [{k: e.get(k) for k in ("id", "skill", "ref", "symptom", "op_family", "arch", "kind", "confidence")}
                for _, e in final]
        print(json.dumps({"hits": len(hits), "entries": hits}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
