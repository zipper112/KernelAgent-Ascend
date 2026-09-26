#!/usr/bin/env python3
"""tools/supervise.py —— 自主迭代监督仪表（人盯 agent 的眼睛）。

读任务 audit.log，按轮分组检查事件序列是否符合协议预期：
  router-query → production-ref|blindspot → skill-inject → candidate-write
  → verify-step → [bench-step] → review
违规类型：
  SKIP-RESEARCH   candidate-write 之前无 router-query（模型直接开写=知识库白费）
  SKIP-SKILL      candidate-write 之前无 skill-inject 或注入数=0（未用知识）
  VERIFY-SKIPPED  review 前无 verify-step
  BENCH-BEFORE-VERIFY
  NO-REVIEW       轮结束无 review
  KNOWLEDGE-UNUSED candidate-write 的 detail.knowledge_used 为空（writer 未引用任何知识）
用法：python tools/supervise.py <task> [--round N]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED = ["router-query", "production-ref|blindspot", "skill-inject",
            "candidate-write", "verify-step", "review"]


def load_events(task: Path, round_: int | None) -> list[dict]:
    audit = task / "docs" / "audit.log"
    if not audit.exists():
        return []
    evs = []
    for line in audit.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("action") in ("router-query", "production-ref", "blindspot", "skill-inject",
                               "candidate-write", "verify-step", "bench-step", "review"):
            if round_ is None or d.get("round") == round_:
                evs.append(d)
    return evs


def check_round(evs: list[dict]) -> list[str]:
    v: list[str] = []
    seq = [e["action"] for e in evs]
    def first(a):
        try:
            return seq.index(a)
        except ValueError:
            return None
    def last(a):
        return len(seq) - 1 - seq[::-1].index(a) if a in seq else None
    cw = first("candidate-write")
    if cw is None:
        return v + ["NO-CANDIDATE（本轮无产出）"]
    if first("router-query") is None or first("router-query") > cw:
        v.append("SKIP-RESEARCH：candidate-write 前无 router-query（模型直接开写）")
    if first("production-ref") is None or first("production-ref") > cw:
        if first("blindspot") is None:
            v.append("SKIP-PRODUCTION：未查生产代码索引且无盲区声明")
    si = [e for e in evs if e["action"] == "skill-inject"]
    # 以本轮最后一个 candidate-write 为锚（历史 repair 数据不产生假违规）
    cw_last = max(i for i, e in enumerate(evs) if e["action"] == "candidate-write")
    si_before = [e for e in si if evs.index(e) < cw_last]
    if not si_before or si_before[-1]["detail"].get("n", 0) == 0:
        if not any(e["action"] == "blindspot" for e in evs):
            v.append("SKIP-SKILL：candidate-write 前无 skill 注入（知识库未用）")
    if first("verify-step") is None:
        v.append("VERIFY-SKIPPED：无 verify")
    elif first("bench-step") is not None and first("bench-step") < first("verify-step"):
        v.append("BENCH-BEFORE-VERIFY")
    elif first("bench-step") is not None and evs[first("verify-step")]["detail"].get("passed") is not True \
            and first("bench-step") > first("verify-step"):
        v.append("BENCH-AFTER-VERIFY-FAIL：verify 未过仍 bench")
    if first("review") is None:
        v.append("NO-REVIEW")
    # writer 是否引用知识
    for e in evs:
        if e["action"] == "candidate-write":
            ku = e.get("detail", {}).get("knowledge_used") or []
            if not ku:
                v.append(f"KNOWLEDGE-UNUSED：{e.get('target')} 的 writer 未声明引用任何知识")
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--round", type=int, default=None)
    args = ap.parse_args()
    task = ROOT / "tasks" / args.task
    evs = load_events(task, args.round)
    if not evs:
        print("无迭代事件（尚未起跑）")
        return 1
    # 分组：candidate-write 开启新轮（round 字段可能缺失——历史事件/RESEARCH 先行的修复前记录）
    # repair：无 round 的事件按时间顺序归入最近的后续轮（出现在首个带 round 的 candidate-write 前 = 该轮 RESEARCH）
    rounds: list[int] = []
    by_round: dict[int, list[dict]] = {}
    pending: list[dict] = []
    next_round = 1
    for e in evs:
        r = e.get("round")
        if e["action"] == "candidate-write":
            rr = r if r is not None else next_round
            if rr not in by_round:
                rounds.append(rr)
                by_round[rr] = []
            by_round[rr].extend(pending)      # 前置 RESEARCH 事件归本轮
            pending = []
            by_round[rr].append(e)
            next_round = rr + 1
        else:
            if r is not None and r in by_round:
                by_round[r].append(e)
            elif r is not None and by_round:
                # 带轮号但轮未开（异常序）：就近挂最后轮
                by_round[rounds[-1]].append(e)
            else:
                pending.append(e)
    if pending:
        if rounds:
            by_round[rounds[-1]].extend(pending)
        else:
            rounds.append(1)
            by_round[1] = pending
    all_ok = True
    for r in rounds:
        viol = check_round(by_round[r])
        seq = " → ".join(e["action"] for e in by_round[r])
        status = "OK " if not viol else "VIOLATION"
        print(f"[{status}] round {r}: {seq[:200]}")
        for x in viol:
            print(f"    !! {x}")
            all_ok = False
    print("\n监督结论：", "全部轮次符合预期（调研→知识→写→验→评）" if all_ok else "存在违规，需人工干预")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
