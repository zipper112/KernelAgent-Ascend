#!/usr/bin/env python3
"""kda —— KDA-Ascend harness CLI（v0.2：批 A+B 实装）。

已实装：version / new-task / status / log / contract(--verify) / budget --report / verify / bench
桩（批 C-E）：gate / promote / ab / diagnose / export / unlock（rc=2 not-implemented，防误判成功）
所有命令统一：audit.log 追加一条 + 会话锁（瞬时持锁）+ state.json 读写（协议 §1/§1a）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from harness.core.evidence import Evidence
from harness.core.state import SessionLock, TaskState

PROTOCOL_DOC = "docs/design/interaction-protocol.md"
VERSION = "0.2.0-min-loop"
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "tasks" / "_template"

STILL_STUBS = ("gate", "promote", "ab", "diagnose", "export")


def _out(data: dict, rc: int) -> int:
    print(json.dumps(data, ensure_ascii=False))
    return rc


def _task_root(task_arg: str) -> Path:
    t = Path(task_arg)
    return (REPO_ROOT / "tasks" / t.name) if not t.is_absolute() else t


def _with_lock_and_state(task_root: Path, fn, mode: str = "companion"):
    """命令公共骨架：锁（O_EXCL/接管）→ state 校验 → fn(ev, lock, state) → audit。"""
    ev = Evidence(task_root)
    lock = SessionLock(task_root, ev, mode=mode)
    try:
        lock.acquire_or_takeover()
    except PermissionError as e:
        ev.log_audit("harness", "lock-refused", detail={"error": str(e)[:150]})
        return _out({"error": "lock-busy", "detail": str(e)}, 2)
    st = TaskState(task_root, ev)
    try:
        state = st.require()
    except FileNotFoundError as e:
        ev.log_audit("harness", "state-missing")
        return _out({"error": str(e)}, 2)
    return fn(ev, lock, st, state)


# ---------- 已实装命令 ----------

def cmd_version(_args) -> int:
    return _out({"version": VERSION, "protocol": PROTOCOL_DOC}, 0)


def cmd_new_task(args) -> int:
    """七件套复制 + state.json 引导（round=0）+ 分支提示（协议 §1 v0.2）。"""
    target = _task_root(args.dir)
    if target.exists() and any(target.iterdir()):
        return _out({"error": "already-exists", "dir": str(target)}, 2)
    target.mkdir(parents=True, exist_ok=True)
    # 复制模板（剥 .gitkeep 语义保留；run/ 下生成 state.json）
    for item in TEMPLATE.iterdir():
        if item.name == "run":
            continue
        dst = target / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    (target / "run").mkdir(exist_ok=True)
    state = TaskState.init_for_new_task(target, target.name)
    ev = Evidence(target)
    ev.log_audit("harness", "new-task", target=str(target), round_=0,
                 detail={"files": "7-piece from _template"})
    return _out({"ok": True, "task": target.name, "round": 0,
                 "next": [f"cd {target}",
                          f"git checkout -b {state['task'] and 'task/' + target.name}",
                          "填 task.yaml 8 槽 + config.yaml + bench/workloads.yaml",
                          "agent: 路由检索 → draft.md → gen-plan → kda contract --lock"]}, 0)


def cmd_status(args) -> int:
    task_root = _task_root(args.task)
    if args.round_only:
        try:
            s = TaskState(task_root, Evidence(task_root)).require()
            print(s["round"])
            return 0
        except FileNotFoundError:
            return _out({"error": "state-missing"}, 2)
    return _with_lock_and_state(task_root, lambda ev, lock, st, state: _out({
        "task": state["task"], "round": state["round"], "mode": state["mode"],
        "phase": state["phase"], "best": state["best"],
        "direction_fails": state["direction_fails"], "stall_count": state["stall_count"],
        "pause": state["pause"], "terminal": state["terminal"],
    }, 0))


def cmd_log(args) -> int:
    task_root = _task_root(args.task)
    audit = task_root / "docs" / "audit.log"
    if not audit.exists():
        return _out({"entries": [], "note": "audit.log 尚无记录"}, 0)
    lines = [l for l in audit.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.actor:
        lines = [l for l in lines if f'"actor": "{args.actor}"' in l]
    if args.tail:
        lines = lines[-args.tail:]
    return _out({"entries": [json.loads(l) for l in lines[-50:]]}, 0)


def cmd_contract(args) -> int:
    """v0.2 批 B：--verify 实装（hash 比对）；--lock/--unlock 留批 C（依赖 plan 流程）。"""
    task_root = _task_root(args.task)
    if args.verify:
        def run(ev, lock, st, state):
            import hashlib
            plan = task_root / "docs" / "plan.md"
            lockf = task_root / "docs" / "plan.md.lock"
            if not lockf.exists():
                return _out({"error": "not-locked", "hint": "kda contract --lock（批 C 实装）"}, 2)
            d = json.loads(lockf.read_text(encoding="utf-8"))
            cur = hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else ""
            ok = cur == d.get("plan_sha")
            ev.log_audit("harness", "contract-verify", detail={"ok": ok})
            return _out({"ok": ok, "plan_sha_match": ok}, 0 if ok else 1)
        return _with_lock_and_state(task_root, run)
    if args.lock or args.unlock:
        return _out({"error": "not-implemented", "command": "contract --lock/--unlock",
                     "phase": "批 C", "spec": f"{PROTOCOL_DOC}#1"}, 2)
    return _out({"error": "需要 --verify（--lock 批 C）"}, 2)


def cmd_budget(args) -> int:
    """§7.2a 自报通道：两级账本追加。"""
    if not args.report:
        return _out({"error": "需要 --report --tokens N --round R"}, 2)
    task_root = _task_root(args.task)

    def run(ev, lock, st, state):
        res = ev.append_usage(task=state["task"], role="host-agent", model="self-report",
                              prompt_tokens=int(args.tokens), completion_tokens=0,
                              source="self-report",
                              global_ledger=REPO_ROOT / "run-global" / "usage.jsonl")
        ev.log_audit("agent", "budget-report", round_=int(args.round),
                     detail={"tokens": int(args.tokens), **res})
        return _out({"ok": True, "round": int(args.round), **res}, 0)
    return _with_lock_and_state(task_root, run)


def _device_contention(task_root: Path, ev, state: dict, device_id: int) -> dict | None:
    """bench 前争用探针：目标卡上有他人进程 → audit 记 warning（数据可能污染，不阻塞——共用机现实）。"""
    try:
        import yaml as _y
        cfg = _y.safe_load((task_root / "config.yaml").read_text(encoding="utf-8"))
        rem = cfg.get("execution", {}).get("remote", {})
        from infra.remote.executor import RemoteExecutor, RemoteTarget
        tgt = RemoteTarget(jump=rem.get("jump", "jump"), host=rem.get("host", "yq-e15"))
        r = RemoteExecutor(tgt).run(
            f"npu-smi info | grep -A6 'NPU {device_id} ' | grep -c 'Process id' || true", timeout_s=30)
        n = int((r.get("stdout") or "0").strip().splitlines()[-1] or 0)
        info = {"device": device_id, "foreign_processes": n}
        if n > 0:
            ev.log_audit("harness", "contention-warning", round_=state.get("round"),
                         detail={**info, "note": "目标卡有占用进程，bench 数据可能污染"})
        return info
    except Exception:   # noqa: BLE001 —— 探针失败不阻塞测量
        return None


def _run_remote_job(task_root: Path, state: dict, kind: str, candidate_id: str,
                    workload_set: str, ev: Evidence) -> dict:
    """verify/bench 公共：JobSpec 组装（含 payload 必备件）→ run_job → evidence 落盘。"""
    import yaml
    from infra.remote.executor import RemoteTarget
    from infra.remote.sync import JobSpec, run_job

    cfg = yaml.safe_load((task_root / "config.yaml").read_text(encoding="utf-8"))
    rem = cfg.get("execution", {}).get("remote", {})
    if not rem.get("enabled", False):
        return {"error": "remote-disabled（本地执行模式批 C 实装）"}
    wls_data = yaml.safe_load((task_root / "bench" / "workloads.yaml").read_text(encoding="utf-8"))
    # workload 透传完整定义（含多张量 inputs spec；dtype 缺省补 fp16）
    wls = []
    for w in wls_data["workloads"]:
        item = {"id": w["id"], "axes": w["axes"], "dtype": w.get("dtype", "fp16")}
        if "inputs" in w:
            item["inputs"] = w["inputs"]
        wls.append(item)
    meas = cfg.get("measurement", {})
    files = [f"solution/{candidate_id}/candidate.py"]
    for extra in ("reference.py", "bench/workloads.yaml"):   # §8b：workload 定义必达（v0.2 修复：漏推致远端读旧残留）
        if (task_root / extra).exists():
            files.append(extra)
    spec = JobSpec(job_id=f"{state['task']}-{kind}-{candidate_id}-{int(time.time())}",
                   kind=kind, candidate_id=candidate_id, task=state["task"],
                   files=files, workloads=wls, workload_set=workload_set,
                   timeout_s=600, device_id=rem.get("device_id", 0),
                   extra={"warmup": meas.get("warmup", 3), "samples": meas.get("samples", 5)})
    target = RemoteTarget(jump=rem.get("jump", "jump"), host=rem.get("host", "yq-e15"),
                          exec_mode=rem.get("exec_mode", "docker"),
                          docker_image=rem.get("docker_image", ""),
                          cann_env=rem.get("cann_env", ""))
    return run_job(target, spec, rem.get("mirror_root", "~/kda-ascend"),
                   "infra/remote/runner.py", task_root, task_root / "results",
                   repo_root=REPO_ROOT)


def cmd_verify(args) -> int:
    """verify：远端四步协议 → evidence 即时落盘（失败也入链计 direction_fails，v0.2）。"""
    task_root = _task_root(args.task)

    def run(ev, lock, st, state):
        r = _run_remote_job(task_root, state, "verify", args.candidate, args.workload_set, ev)
        if not r.get("ok"):
            ev.log_audit("harness", "kda verify", target=args.candidate,
                         round_=state["round"], detail={"error": str(r.get("error"))[:150]})
            return _out({"error": r.get("error", "job-failed"), "stage": r.get("stage")}, 2)
        res = r.get("result", {})
        passed = bool(res.get("passed"))
        direction = _current_direction(task_root, state["round"], ev=ev, candidate_id=args.candidate)
        if passed:
            ev.append_solution(args.candidate, parent_id=_parent_of(ev, args.candidate),
                               direction=direction or "baseline",
                               hypothesis="", status="keep", round_=state["round"], stage="verify")
        else:
            # v0.2：verify 失败入链 + 计数（封死单方向无限烧）
            ev.append_solution(args.candidate, parent_id=_parent_of(ev, args.candidate),
                               direction=direction or "unknown",
                               hypothesis="", status="reject", round_=state["round"], stage="verify")
            if direction:
                fails = st.bump_direction_fail(direction)
                ev.log_audit("harness", "fuse-check", target=f"direction={direction}",
                             round_=state["round"], detail={"consecutive_fails": fails})
        ev.log_audit("agent", "kda verify", target=args.candidate, round_=state["round"],
                     detail={"passed": passed, "err_ratio": [w.get("err_ratio") for w in res.get("workloads", [])]})
        return _out({"passed": passed, "workloads": res.get("workloads", []),
                     "workload_set": args.workload_set}, 0 if passed else 1)
    return _with_lock_and_state(task_root, run)


def _current_direction(task_root: Path, round_: int, ev: "Evidence | None" = None, candidate_id: str = "") -> str | None:
    """direction 三源回退：①round-N-contract（陪伴模式 agent 手写）②该候选最近一次
    candidate-write 的 audit detail（自主循环路径）③baseline。熔断②依赖此值。"""
    rc = task_root / "run" / f"round-{round_}-contract.md"
    if rc.exists():
        for line in rc.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("direction:"):
                return line.split("direction:", 1)[1].strip()
    if ev is not None and (task_root / "docs" / "audit.log").exists():
        import json as _json
        last = None
        for line in (task_root / "docs" / "audit.log").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = _json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("action") == "candidate-write" and d.get("detail", {}).get("direction"):
                if not candidate_id or d.get("target") == candidate_id:
                    last = d["detail"]["direction"]
        return last
    return None


def _parent_of(ev: Evidence, candidate_id: str) -> str | None:
    """bench-auto 行的 parent 语义：无法从调用序可靠推断（v0.2 诚实化——此前取"最近其他候选"
    会把先出现的候选挂到后出现的下面，破坏 DAG）。返回 None；真正的 DAG 边由 agent 在
    solutions.jsonl 补写或 gate 校验时显式声明。"""
    return None


def cmd_bench(args) -> int:
    """bench：远端交错采样 → benchmark.csv 行 + solutions.jsonl 行 + audit（v0.2 即时落盘）。"""
    task_root = _task_root(args.task)

    def run(ev, lock, st, state):
        import yaml as _y
        _rem = _y.safe_load((task_root / "config.yaml").read_text(encoding="utf-8")) \
                  .get("execution", {}).get("remote", {})
        contention = _device_contention(task_root, ev, state, _rem.get("device_id", 0))
        r = _run_remote_job(task_root, state, "bench", args.candidate, args.workload_set, ev)
        if not r.get("ok"):
            ev.log_audit("harness", "kda bench", target=args.candidate,
                         round_=state["round"], detail={"error": str(r.get("error"))[:150]})
            return _out({"error": r.get("error", "job-failed"), "stage": r.get("stage")}, 2)
        res = r.get("result", {})
        wls = res.get("workloads", [])
        mean = sum(w.get("mean_us", 0) for w in wls) / max(len(wls), 1)
        p50 = sorted(w.get("p50_us", 0) for w in wls)[len(wls) // 2] if wls else None
        p99 = max((w.get("p99_us", 0) for w in wls), default=None)
        speedup = (sum(w.get("speedup_vs_ref", 0) for w in wls) / len(wls)
                   if wls and "speedup_vs_ref" in wls[0] else None)
        ev.append_benchmark(args.candidate, None, state["phase"],
                            args.workload_set, mean, p50, p99, speedup, verdict="keep",
                            note="bench-auto" + (" [contention!]" if contention and contention.get("foreign_processes") else ""))
        ev.log_audit("agent", "kda bench", target=args.candidate, round_=state["round"],
                     detail={"mean_us": round(mean, 1) if mean else None,
                             **({"contention": contention} if contention else {})})
        return _out({"ok": True, "mean_us": mean, "p50_us": p50, "p99_us": p99,
                     "speedup": speedup, "workloads": wls,
                     **({"contention": contention} if contention else {})}, 0)
    return _with_lock_and_state(task_root, run)


# ---------- 桩命令 ----------

def cmd_stub(name: str):
    def _fn(args) -> int:
        task = getattr(args, "task", None)
        if task:
            task_root = _task_root(task)
            if task_root.exists():
                Evidence(task_root).log_audit("harness", f"kda {name}", detail={"stub": True})
        return _out({"error": "not-implemented", "command": name, "phase": "批 C-E",
                     "spec": f"{PROTOCOL_DOC}#1"}, 2)
    return _fn


def main() -> int:
    p = argparse.ArgumentParser(prog="kda", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version").set_defaults(fn=cmd_version)

    sp = sub.add_parser("new-task"); sp.add_argument("dir")
    sp.set_defaults(fn=cmd_new_task)

    sp = sub.add_parser("status"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--round", dest="round_only", action="store_true")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("log"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--tail", type=int); sp.add_argument("--actor")
    sp.set_defaults(fn=cmd_log)

    sp = sub.add_parser("contract"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--lock", action="store_true"); sp.add_argument("--unlock", action="store_true")
    sp.add_argument("--verify", action="store_true"); sp.add_argument("--reason")
    sp.set_defaults(fn=cmd_contract)

    sp = sub.add_parser("budget"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--report", action="store_true")
    sp.add_argument("--tokens", type=int); sp.add_argument("--round", type=int, default=0)
    sp.add_argument("--note")
    sp.set_defaults(fn=cmd_budget)

    sp = sub.add_parser("verify"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--candidate", required=True)
    sp.add_argument("--workload-set", default="l0", choices=["l0", "l1", "full"])
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("bench"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--candidate", required=True)
    sp.add_argument("--workload-set", default="l0", choices=["l0", "l1", "full"])
    sp.set_defaults(fn=cmd_bench)

    for name in STILL_STUBS:
        sp = sub.add_parser(name); sp.add_argument("--task", default="rmsnorm-smoke")
        sp.set_defaults(fn=cmd_stub(name))

    sp = sub.add_parser("unlock"); sp.add_argument("--task", default="rmsnorm-smoke")
    sp.add_argument("--stale", action="store_true")
    sp.set_defaults(fn=cmd_stub("unlock"))

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
