"""harness/control/memory.py —— 迭代记忆层（Humanize RLCR 状态存续 + KDA 证据驱动 + 比赛经验沉淀）。

三源机制的内化（补课 2026-09-26，用户裁定：此前循环每轮失忆=裸奔）：
- Humanize：每轮注入 上轮评审全文 + 契约 + 候选历史 —— 跨轮状态存续；
- KDA：bench 证据（数值+形状级）驱动下一轮假设 —— 证据进 prompt 不进直觉；
- 比赛：BitLesson 经验沉淀（每轮一条，失败教训与成功手法都留）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class IterationMemory:
    """一个任务的迭代记忆：轮档案 + 经验库。落盘 run/memory/（gitignored，摘要入 audit）。"""

    def __init__(self, task_root: Path):
        self.task = Path(task_root)
        self.dir = self.task / "run" / "memory"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lessons = self._load_lessons()

    # ---------- 轮档案（Humanize：上轮评审全文与证据进下轮上下文） ----------

    def save_round(self, round_: int, record: dict) -> None:
        """轮档案：candidate 代码快照 + verify/bench 数值 + 评审原文 + 方向。
        下轮组装 prompt 时 load_last_round 全文注入（不是摘要——评审措辞里有修复线索）。"""
        p = self.dir / f"round-{round_}.json"
        p.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    def load_round(self, round_: int) -> dict | None:
        p = self.dir / f"round-{round_}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def last_round(self) -> tuple[int, dict] | None:
        rounds = sorted((int(m.group(1)) for m in
                         (re.match(r"round-(\d+)\.json", p.name) for p in self.dir.glob("round-*.json"))
                         if m))
        if not rounds:
            return None
        r = rounds[-1]
        return r, self.load_round(r)

    def trend_digest(self, n: int = 3) -> str:
        """P1-4 近 N 轮趋势表（direction × mean_us × verdict 一行一轮）——
        writer 一眼看出"哪个方向在收敛、哪个在原地打转"；全量明细仍在证据账本。"""
        rounds = sorted((int(m.group(1)) for m in
                         (re.match(r"round-(\d+)\.json", p.name) for p in self.dir.glob("round-*.json"))
                         if m))
        if not rounds:
            return "（尚无轮档案）"
        lines = []
        for r in rounds[-n:]:
            rec = self.load_round(r) or {}
            b = rec.get("bench") or {}
            mean = b.get("mean_us")
            lines.append(f"- round {r}: dir={rec.get('direction')} "
                         f"mean={f'{mean:.0f}us' if mean else 'N/A'} "
                         f"verify={'过' if (rec.get('verify') or {}).get('passed') else '挂'}")
        return "\n".join(lines)

    # ---------- 证据账本（KDA：数值级证据驱动假设） ----------

    def evidence_digest(self, best_us: float | None) -> str:
        """全部 bench 历史 + 当前最优的紧凑表（进 writer prompt 的证据段）。"""
        csv = self.task / "docs" / "benchmark.csv"
        if not csv.exists():
            return "（尚无 bench 证据）"
        rows = []
        for line in csv.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 8 and parts[5] not in ("n/a", "", "0.0"):
                rows.append(f"{parts[0][11:]} {parts[1]} mean={parts[5]}us p50={parts[6]}us")
        if not rows:
            return "（尚无有效 bench 行）"
        head = "\n".join(rows[-10:])
        best = f"\n当前历史最优 mean_us={best_us}" if best_us else ""
        return f"最近 bench 证据（时间倒序最近 10 条）：\n{head}{best}"

    # ---------- BitLesson（比赛：经验沉淀辅助下一次迭代） ----------

    def _load_lessons(self) -> list[dict]:
        p = self.dir / "lessons.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def add_lesson(self, round_: int, kind: str, text: str) -> None:
        """kind ∈ {win, fail, insight}；每轮至多一条（Humanize 硬校验⑧的同款纪律）。"""
        self.lessons.append({"round": round_, "kind": kind, "text": text[:300]})
        (self.dir / "lessons.json").write_text(
            json.dumps(self.lessons, ensure_ascii=False, indent=1), encoding="utf-8")

    def lessons_digest(self, n: int = 5) -> str:
        """P1-7 选课注入（Humanize bitlesson-selector 的规则版）：win 优先 + 时间近者优先，
        cap n 条替代全量（防经验库膨胀淹没 prompt）。"""
        if not self.lessons:
            return "（尚无沉淀经验）"
        ranked = sorted(self.lessons,
                        key=lambda l: (l.get("kind") == "win", l.get("round", 0)),
                        reverse=True)[:n]
        return "\n".join(f"- [r{l['round']}{l['kind']}] {l['text']}" for l in ranked)
