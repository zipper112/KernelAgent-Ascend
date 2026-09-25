"""harness/core/evidence.py —— 证据链唯一写方（协议 §2/§3）。

benchmark.csv / solutions.jsonl / audit.log 只有本模块可写（反作弊地基）；
所有写入 append-only、逐条落盘（无缓冲），崩溃不产生半行。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

CSV_HEADER = "ts,candidate_id,parent_id,phase,workload_set,mean_us,p50_us,p99_us,speedup,verdict,note"


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Evidence:
    """一个任务的证据三件套写入器。task_root = tasks/<task>/。"""

    def __init__(self, task_root: Path):
        self.root = Path(task_root)
        self.docs = self.root / "docs"
        self.docs.mkdir(parents=True, exist_ok=True)
        self.csv = self.docs / "benchmark.csv"
        self.jsonl = self.docs / "solutions.jsonl"
        self.audit = self.docs / "audit.log"

    # ---------- audit（所有命令共用的审计根） ----------

    def log_audit(self, actor: str, action: str, target: str = "", round_: int | None = None,
                  detail: dict | None = None) -> None:
        """协议 §3.4：{ts, actor, action, target, round, detail}；actor∈{agent,gate,human,harness}。"""
        assert actor in ("agent", "gate", "human", "harness"), f"非法 actor: {actor}"
        rec = {"ts": _ts(), "actor": actor, "action": action, "target": target,
               "round": round_, "detail": detail or {}}
        with open(self.audit, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- benchmark.csv（§3.1） ----------

    def _ensure_csv_header(self) -> None:
        if not self.csv.exists() or self.csv.stat().st_size == 0:
            self.csv.write_text(CSV_HEADER + "\n", encoding="utf-8")

    def append_benchmark(self, candidate_id: str, parent_id: str | None, phase: str,
                         workload_set: str, mean_us: float | None, p50_us: float | None,
                         p99_us: float | None, speedup: float | None, verdict: str,
                         note: str = "") -> None:
        """verdict ∈ {keep, revise, reject}；speedup 相对锁定 baseline（无则 n/a）。"""
        assert verdict in ("keep", "revise", "reject"), f"非法 verdict: {verdict}"
        assert workload_set in ("l0", "l1", "full"), f"非法 workload_set: {workload_set}"
        self._ensure_csv_header()

        def num(v):
            return "n/a" if v is None else f"{v:.1f}"
        row = (f"{_ts()},{candidate_id},{parent_id or ''},{phase},{workload_set},"
               f"{num(mean_us)},{num(p50_us)},{num(p99_us)},{num(speedup)},"
               f"{verdict},\"{note}\"")
        with open(self.csv, "a", encoding="utf-8", newline="") as f:
            f.write(row + "\n")

    # ---------- solutions.jsonl（§3.2 候选 DAG） ----------

    def append_solution(self, candidate_id: str, parent_id: str | None, direction: str,
                        hypothesis: str, status: str, round_: int,
                        fallback: bool = False, stage: str = "bench") -> None:
        """status ∈ {keep, revise, reject}；verify 失败也入链（v0.2：stage=verify 计 direction_fails）。"""
        assert status in ("keep", "revise", "reject"), f"非法 status: {status}"
        rec = {"candidate_id": candidate_id, "parent_id": parent_id, "direction": direction,
               "hypothesis": hypothesis, "diff_ref": f"solution/{candidate_id}/",
               "evidence_refs": [f"benchmark.csv#{candidate_id}"], "status": status,
               "round": round_, "fallback": fallback, "stage": stage}
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_solutions(self) -> list[dict]:
        if not self.jsonl.exists():
            return []
        return [json.loads(l) for l in self.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---------- usage 账本（§7.2/§7.2a） ----------

    def append_usage(self, task: str, role: str, model: str, prompt_tokens: int,
                     completion_tokens: int, source: str = "api",
                     global_ledger: Path | None = None) -> dict:
        """两级账本追加；返回 {cumulative_global, cumulative_task}。"""
        task_ledger = self.root / "run" / "usage.jsonl"
        task_ledger.parent.mkdir(parents=True, exist_ok=True)
        cum_task = self._cumulative(task_ledger) + prompt_tokens + completion_tokens
        rec = {"ts": _ts(), "task": task, "role": role, "model": model,
               "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
               "cumulative_task": cum_task, "source": source}
        with open(task_ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if global_ledger is not None:
            global_ledger.parent.mkdir(parents=True, exist_ok=True)
            cum_g = self._cumulative(global_ledger) + prompt_tokens + completion_tokens
            grec = dict(rec)
            grec["cumulative_global"] = cum_g
            del grec["cumulative_task"]
            with open(global_ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(grec, ensure_ascii=False) + "\n")
            return {"cumulative_global": cum_g, "cumulative_task": cum_task}
        return {"cumulative_task": cum_task}

    @staticmethod
    def _cumulative(ledger: Path) -> int:
        if not ledger.exists():
            return 0
        last = ""
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = line
        if not last:
            return 0
        d = json.loads(last)
        return d.get("cumulative_global", d.get("cumulative_task", 0))


# 原子写 helper（state.json 用；os.replace 同文件系统原子）
def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
