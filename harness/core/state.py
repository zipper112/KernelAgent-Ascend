"""harness/core/state.py —— 会话锁 + state.json 生命周期（协议 §1a/§1b v0.2）。

零 LLM（core 铁律）。锁 = run/lock（O_EXCL 原子创建，30 分钟陈旧自动接管）；
state.json = 循环状态机唯一载体（写方=锁持有进程）。
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from .evidence import Evidence, atomic_write_json

STALE_MINUTES = 30

# 伪方向（P0-4）：parse/接口失败的占位方向不得计入 direction_fails（污染熔断②），
# 也不得作为 candidates 历史的有效方向呈现
PSEUDO_DIRECTIONS = {"", "parse-failed", "unspecified", "unknown", "none"}


def is_pseudo_direction(direction: str | None) -> bool:
    return str(direction or "").strip().lower() in PSEUDO_DIRECTIONS


class SessionLock:
    """瞬时持锁：with 块内校验+刷新，退出不删（陪伴模式语义）。"""

    def __init__(self, task_root: Path, evidence: Evidence, mode: str = "companion"):
        self.run = Path(task_root) / "run"
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = self.run / "lock"
        self.ev = evidence
        self.mode = mode
        self.took_over = False

    def acquire_or_takeover(self) -> dict:
        """返回锁内容 dict。活跃异 mode → raise PermissionError（调用方转码 2）。"""
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        content = {"mode": self.mode, "host": socket.gethostname(), "created_at": now, "last_seen": now}
        if self.lock.exists():
            old = json.loads(self.lock.read_text(encoding="utf-8"))
            age_min = (time.time() - self.lock.stat().st_mtime) / 60
            if age_min >= STALE_MINUTES:
                self.took_over = True
                self._write(content)
                self.ev.log_audit("harness", "lock-takeover", target="run/lock",
                                  detail={"old_mode": old.get("mode"), "age_min": round(age_min, 1)})
                return content
            if old.get("mode") != self.mode:
                raise PermissionError(f"活跃锁 mode={old.get('mode')}（last_seen {age_min:.0f} 分钟前）")
            content = {**old, "last_seen": now}
            self._write(content)
            return content
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # 原子创建
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))
        return content

    def _write(self, content: dict) -> None:
        tmp = self.lock.with_suffix(".tmp")
        tmp.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.lock)

    def release(self) -> None:
        self.lock.unlink(missing_ok=True)


class TaskState:
    """state.json 读写（协议 §1b schema v1）。"""

    def __init__(self, task_root: Path, evidence: Evidence):
        self.path = Path(task_root) / "run" / "state.json"
        self.ev = evidence

    @staticmethod
    def init_for_new_task(task_root: Path, task_name: str) -> dict:
        """new-task 引导（v0.2：round=0, mode=companion 缺省）。"""
        path = Path(task_root) / "run" / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"schema": 1, "task": task_name, "mode": "companion", "round": 0,
                 "best": None, "direction_fails": {}, "stall_count": 0,
                 "writer_fails": 0, "last_progress": None,
                 "last_verdict": None, "pause": None, "terminal": None,
                 "phase": "P1", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        atomic_write_json(path, state)
        return state

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def require(self) -> dict:
        """缺失即错（v0.2：不自动重建，防静默重置计数）。"""
        s = self.load()
        if s is None:
            raise FileNotFoundError("state-missing: run kda new-task or restore from git")
        return s

    def update(self, **fields) -> dict:
        s = self.require()
        s.update(fields)
        s["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(self.path, s)
        return s

    def bump_direction_fail(self, direction: str) -> int:
        """verify/bench reject 时 +1；返回该方向累计。达到 3 由调用方触发换向注入。
        P0-4：伪方向（parse-failed 等）直接返回 0 不计数。"""
        if is_pseudo_direction(direction):
            return 0
        s = self.require()
        fails = s.get("direction_fails", {})
        fails[direction] = fails.get(direction, 0) + 1
        self.update(direction_fails=fails)
        return fails[direction]

    def bump_writer_fail(self) -> int:
        """P0-4：writer 失败（parse-fail/preflight 拒绝且修复无效）独立计数。
        连续 3 次 = 模型/协议问题（非方向问题），由调用方转 pause 终态。"""
        s = self.require()
        n = int(s.get("writer_fails", 0)) + 1
        self.update(writer_fails=n)
        return n

    def reset_writer_fails(self) -> None:
        if self.require().get("writer_fails"):
            self.update(writer_fails=0)

    def bump_stall(self, progress: str) -> int:
        """P0-3 漂移状态机：progress ∈ {ADVANCED, STALLED, REGRESSED}（机器判定，
        不采信 LLM 自评）。ADVANCED 清零，否则累计；返回当前 stall_count。"""
        assert progress in ("ADVANCED", "STALLED", "REGRESSED"), f"非法 progress: {progress}"
        s = self.require()
        n = 0 if progress == "ADVANCED" else int(s.get("stall_count", 0)) + 1
        self.update(stall_count=n, last_progress=progress)
        return n
