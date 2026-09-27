#!/usr/bin/env python3
"""k8-triton bench —— 性能测量（零 LLM；canonical runner：L2 清除+交错采样）。

用法：
  python bench.py --solution solution/cXXX/candidate.py [--full] [--record]
  默认 dev 档；--full 全档。--record 追加 benchmark.csv（不传只打印）。
rc=0 出数 / rc=2 执行失败。同机 baseline 由 runner 内部对照测量。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
from verify import HOST, WS, DEVICE, IMAGE, ENTRY_REL, RUNNER_REL, load_workloads, sync_payload  # noqa: E402

REPO = TASK.parent.parent
CSV = TASK / "docs" / "benchmark.csv"
CSV_HEADER = "ts,candidate_id,parent_id,phase,workload_set,mean_us,p50_us,p99_us,speedup,verdict,note"


def run_bench(job: dict) -> dict:
    jpath = TASK / "run" / "job-bench.json"
    jpath.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    subprocess.run(f"cat {jpath} | ssh -o BatchMode=yes {HOST} 'cat > {WS}/payload/job.json'",
                   shell=True, capture_output=True, timeout=60)
    dev = job["physical_device_id"]
    cmd = (f"sudo docker run --rm --device /dev/davinci{dev} --device /dev/davinci_manager "
           f"--device /dev/devmm_svm --device /dev/hisi_hdc "
           f"-v {WS}/payload:/work/payload:ro -v {WS}/results:/work/results "
           f"-v /usr/local/Ascend/driver:/usr/local/Ascend/driver_host:ro "
           f"--entrypoint bash {IMAGE} /work/payload/{ENTRY_REL} "
           f"/work/payload/{RUNNER_REL} --job job.json --results-dir /work/results")
    ex = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd],
                        capture_output=True, text=True, timeout=900)
    cat = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST,
                          f"cat {WS}/results/{job['job_id']}.json"],
                         capture_output=True, text=True, timeout=120)
    if cat.returncode != 0 or not cat.stdout.strip():
        print(f"bench exec rc={ex.returncode}\nstdout tail: {ex.stdout[-400:]}\n"
              f"stderr tail: {ex.stderr[-400:]}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(cat.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--full", action="store_true", help="full workload set")
    ap.add_argument("--record", action="store_true", help="append benchmark.csv")
    args = ap.parse_args()

    p = Path(args.solution)
    cid_dir = p.parent.name if p.is_file() else p.name
    workloads = load_workloads(None, fast=not args.full)
    wset = "full" if args.full else "l0"
    job = {"job_id": f"k8-triton-bench-{cid_dir}-{int(time.time())}",
           "kind": "bench", "candidate_id": cid_dir, "task": TASK.name,
           "workloads": workloads, "device_id": 0, "physical_device_id": DEVICE,
           "extra": {"warmup": 3, "samples": 5}}

    # 争用自查（共用机纪律）
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST,
                        f"npu-smi info | grep -A6 'NPU {DEVICE} ' | grep -c 'Process id' || true"],
                       capture_output=True, text=True, timeout=30)
    try:
        foreign = int((r.stdout or "0").strip().splitlines()[-1] or 0)
    except ValueError:
        foreign = 0
    if foreign > 0:
        print(f"WARNING: NPU {DEVICE} 有 {foreign} 个他人进程——数据可能污染，勿作 keep 依据")

    sync_payload(cid_dir)
    result = run_bench(job)
    wls = result.get("workloads", [])
    if not wls:
        print("no bench results", file=sys.stderr)
        return 2
    mean = sum(w.get("mean_us", 0) for w in wls) / len(wls)
    p50 = sorted(w.get("p50_us", 0) for w in wls)[len(wls) // 2]
    p99 = max((w.get("p99_us", 0) for w in wls), default=0)
    speedup = (sum(w.get("speedup_vs_ref", 0) for w in wls) / len(wls)
               if "speedup_vs_ref" in wls[0] else None)
    unstable = [w["id"] for w in wls if w.get("ref_unstable")]

    print(f"solution: {cid_dir}  set: {wset}  n: {len(wls)}")
    for w in wls:
        print(f"  {w['id']}: mean={w.get('mean_us', 0):.1f}us p50={w.get('p50_us', 0):.1f} "
              f"p99={w.get('p99_us', 0):.1f} ref={w.get('baseline_mean_us', '-')}"
              f"{' [ref-unstable!]' if w.get('ref_unstable') else ''}")
    print(f"\nmean: {mean:.1f}us  p50: {p50:.1f}  p99: {p99:.1f}"
          + (f"  speedup_vs_ref: {speedup:.2f}x" if speedup else ""))
    if unstable:
        print(f"ref-unstable rows (不可作 keep 依据): {unstable}")

    if args.record:
        CSV.parent.mkdir(exist_ok=True)
        new = not CSV.exists()
        with open(CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_HEADER.split(","))
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), cid_dir, "", "bench", wset,
                        f"{mean:.1f}", f"{p50:.1f}", f"{p99:.1f}",
                        f"{speedup:.2f}" if speedup else "n/a", "benched",
                        "bench-auto" + (" [contention!]" if foreign else "")])
        print(f"recorded -> {CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
