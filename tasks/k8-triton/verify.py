#!/usr/bin/env python3
"""k8-triton verify —— 正确性裁判（模仿 flashinfer-contest verify.py；零 LLM，纯规则）。

用法：
  python verify.py --solution solution/cXXX/candidate.py [--fast] [--workload-uuid w02]
  --fast   = dev 集（默认 w02）+ chained 3 步
  默认     = full 集（w01/w02/w03）+ chained 3 步
rc=0 全过 / rc=1 有失败（stdout 逐 workload 状态表，报错全文输出）。
底层经 canonical runner 在 e15 NPU 容器执行（协议不变：chained 终态门）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parent.parent
RUNNER_REL = "infra/remote/runner.py"
ENTRY_REL = "infra/remote/container_entry.sh"

HOST = "yq-e15"
WS = "~/kda-ascend/tasks/k8-triton"
DEVICE = 7
IMAGE = "quay.io/ascend/vllm-ascend:nightly-main"
TOL = {"fp16": 0.004, "bf16": 0.03, "int8": 0.01, "float16": 0.004, "bfloat16": 0.03}


def load_workloads(only: str | None, fast: bool) -> list[dict]:
    import yaml
    data = yaml.safe_load((TASK / "bench" / "workloads.yaml").read_text(encoding="utf-8"))
    wls = data["workloads"]
    if only:
        wls = [w for w in wls if w["id"] == only]
    elif fast:
        # 尊重 task.yaml contract.workload_sets.dev 声明（无声明才退化为首个 repr）
        dev_ids = None
        try:
            tc = yaml.safe_load((TASK / "task.yaml").read_text(encoding="utf-8"))
            dev_ids = (tc.get("contract", {}).get("workload_sets", {}) or {}).get("dev")
        except Exception:
            pass
        if isinstance(dev_ids, list) and dev_ids:
            keep = set(map(str, dev_ids))
            wls = [w for w in wls if str(w["id"]) in keep]
        else:
            wls = [w for w in wls if w.get("repr")][:1] or wls[:1]
    out = []
    for w in wls:
        item = {"id": w["id"], "axes": w["axes"], "dtype": w.get("dtype", "fp16")}
        if "inputs" in w:
            item["inputs"] = w["inputs"]
        out.append(item)
    return out


def sync_payload(cid_dir: str) -> None:
    frozen = [f for f in ("reference.py", "bench/workloads.yaml") if (TASK / f).exists()]
    tar_args = (f"-C {REPO} {RUNNER_REL} {ENTRY_REL} "
                f"-C {TASK} solution/{cid_dir}/candidate.py " + " ".join(frozen))
    r = subprocess.run(
        f"tar cf - {tar_args} | ssh -o BatchMode=yes {HOST} "
        f"'mkdir -p {WS}/payload {WS}/results && tar xf - -C {WS}/payload'",
        shell=True, capture_output=True, text=True, timeout=300, cwd=str(REPO))
    if r.returncode != 0:
        print(f"payload sync failed: {r.stderr[:200]}", file=sys.stderr)
        raise SystemExit(2)


def run_remote(job: dict) -> dict:
    jpath = TASK / "run" / "job-verify.json"
    jpath.parent.mkdir(exist_ok=True)
    jpath.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    subprocess.run(
        f"cat {jpath} | ssh -o BatchMode=yes {HOST} 'cat > {WS}/payload/job.json'",
        shell=True, capture_output=True, timeout=60)
    dev = job["physical_device_id"]
    cmd = (f"sudo docker run --rm --device /dev/davinci{dev} --device /dev/davinci_manager "
           f"--device /dev/devmm_svm --device /dev/hisi_hdc "
           f"-v {WS}/payload:/work/payload:ro -v {WS}/results:/work/results "
           f"-v /usr/local/Ascend/driver:/usr/local/Ascend/driver_host:ro "
           f"--entrypoint bash {IMAGE} /work/payload/{ENTRY_REL} "
           f"/work/payload/{RUNNER_REL} --job job.json --results-dir /work/results")
    ex = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd],
                        capture_output=True, text=True, timeout=720)
    cat = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST,
                          f"cat {WS}/results/{job['job_id']}.json"],
                         capture_output=True, text=True, timeout=120)
    if cat.returncode != 0 or not cat.stdout.strip():
        print(f"runner exec rc={ex.returncode}\nstdout tail: {ex.stdout[-500:]}\n"
              f"stderr tail: {ex.stderr[-500:]}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(cat.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", required=True, help="path to candidate.py (or its dir)")
    ap.add_argument("--fast", action="store_true", help="dev set + short config")
    ap.add_argument("--workload-uuid", default=None)
    args = ap.parse_args()

    p = Path(args.solution)
    cid_dir = p.parent.name if p.is_file() else p.name
    workloads = load_workloads(args.workload_uuid, args.fast)
    if not workloads:
        print("no workloads selected", file=sys.stderr)
        return 2
    job = {"job_id": f"k8-triton-verify-{cid_dir}-{int(time.time())}",
           "kind": "verify", "candidate_id": cid_dir, "task": TASK.name,
           "workloads": workloads, "device_id": 0, "physical_device_id": DEVICE,
           "extra": {"verify_mode": "chained", "chain_steps": 3}}

    sync_payload(cid_dir)
    result = run_remote(job)

    print(f"solution:   {cid_dir}")
    print(f"mode:       chained x3 (dev={args.fast}, n={len(workloads)})")
    passed = 0
    for w in result.get("workloads", []):
        ok = bool(w.get("passed"))
        passed += ok
        err = w.get("error")
        if ok:
            print(f"  {w['id']}: PASS  steps={w.get('steps')} "
                  f"state_inputs={w.get('state_inputs')} err_ratio={w.get('err_ratio')}")
        else:
            print(f"  {w['id']}: FAIL")
            if err:
                print(f"    error: {err[:800]}")
            for k in ("gate", "step", "err_ratio", "limit"):
                if w.get(k) is not None:
                    print(f"    {k}: {w[k]}")
    total = len(workloads)
    print(f"\npassed: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
