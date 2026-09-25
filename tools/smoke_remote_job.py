#!/usr/bin/env python3
"""tools/smoke_remote_job.py —— Phase 0 NPU 侧验收：rmsnorm-smoke 经 run_job 端到端出数。

链路：本地组 JobSpec → push(payload 两源合流) → e15 docker 容器（CANN 8.2.RC1+torch_npu
2.1.0，设备直通+宿主驱动挂载）执行 runner → pull result 回 tasks/<task>/results/。
用法：python tools/smoke_remote_job.py [--task rmsnorm-smoke] [--kinds verify,bench]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.remote.executor import RemoteTarget  # noqa: E402
from infra.remote.sync import JobSpec, run_job  # noqa: E402


def load_workloads(task_root: Path) -> list[dict]:
    import yaml
    data = yaml.safe_load((task_root / "bench" / "workloads.yaml").read_text(encoding="utf-8"))
    wls = []
    for w in data["workloads"]:
        wls.append({"id": w["id"], "axes": w["axes"], "dtype": w.get("dtype", "fp16")})
    return wls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rmsnorm-smoke")
    ap.add_argument("--kinds", default="verify,bench")
    ap.add_argument("--device", type=int, default=None)   # CLI > config.execution.remote.device_id
    ap.add_argument("--image", default=None)              # CLI > config.execution.remote.docker_image
    ap.add_argument("--mirror-root", default="~/kda-ascend")
    args = ap.parse_args()

    task_root = ROOT / "tasks" / args.task
    cfg = {}
    import yaml
    cfg = yaml.safe_load((task_root / "config.yaml").read_text(encoding="utf-8"))
    rem = cfg.get("execution", {}).get("remote", {})
    target = RemoteTarget(
        jump=rem.get("jump", "jump"), host=rem.get("host", "yq-e15"),
        exec_mode=rem.get("exec_mode", "docker"),
        docker_image=args.image or rem.get("docker_image") or "xllm:glm-clone",
        cann_env=rem.get("cann_env", ""))
    device = args.device if args.device is not None else rem.get("device_id", 0)
    wls = load_workloads(task_root)
    out_dir = task_root / "results"
    rc = 0
    for kind in args.kinds.split(","):
        spec = JobSpec(job_id=f"{args.task}-{kind}-001", kind=kind, candidate_id="c001",
                       task=args.task,
                       files=["solution/c001/candidate.py", "reference.py"],
                       workloads=wls, workload_set="l0",
                       timeout_s=420, device_id=device,
                       extra={"warmup": 3, "samples": 5})
        r = run_job(target, spec, args.mirror_root, "infra/remote/runner.py",
                    task_root, out_dir, repo_root=ROOT)
        compact = {k: v for k, v in r.items() if k != "result"}
        print(f"[{kind}] meta={json.dumps(compact, ensure_ascii=False)[:300]}")
        if r.get("ok") and "result" in r:
            res = r["result"]
            if kind == "verify":
                for w in res.get("workloads", []):
                    print(f"  verify {w['id']}: passed={w.get('passed')} err_ratio={w.get('err_ratio')}")
                print(f"  VERIFY-TOTAL: passed={res.get('passed')} elapsed={res.get('elapsed_s')}s")
            else:
                for w in res.get("workloads", []):
                    line = f"  bench {w['id']}: p50={w.get('p50_us', 0):.1f}us mean={w.get('mean_us', 0):.1f}us"
                    if "speedup_vs_ref" in w:
                        line += f" ref={w.get('baseline_mean_us', 0):.1f}us speedup={w['speedup_vs_ref']:.2f}x"
                    print(line)
        else:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
