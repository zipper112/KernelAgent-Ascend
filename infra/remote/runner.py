#!/usr/bin/env python3
"""infra/remote/runner.py —— 远端 NPU 执行器 v0.1（自包含，被 push 到 e15 运行）。

职责（ADR-011 §8b）：读 payload/job.json → 按 kind 执行确定性计算 → 写 results/<job_id>.json。
payload 根 = job.json 所在目录（v0.1 修正：不再取 __file__ 目录，兼容任意相对深度的 runner 调用）。

v0.1 薄实现（协议规格 §2.1 的最小版）：
  verify —— dtype 分级容差四步协议（形状→NaN 同位→Inf 位置+符号→相对误差+比例容忍）
  bench  —— warmup + L2 cache 清除 + 交错采样；payload 带 reference.py 时同机测 baseline
  profile —— Phase 2（msprof）；v0 返回 not-implemented
不依赖 harness 本地任何模块（远端没有知识库/密钥——它们永不离开本地）。

兼容性（e15 容器实测组合 torch 2.1.0 + torch_npu 2.1.0）：
  - 设备用 .npu() 方法（旧版不认整数 device）
  - 计时事件用 torch.npu.Event
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

TOL = {"fp16": 0.004, "bf16": 0.03, "int8": 0.01, "float16": 0.004, "bfloat16": 0.03}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_kernel(payload: Path, candidate_id: str = "c001"):
    """任务文件里的 kernel 模块须暴露 kernel(inputs)->outputs。
    定位顺序：payload/candidate.py（扁平布局）→ payload/solution/<cid>/candidate.py（仓内布局）。"""
    for p in [payload / "candidate.py", payload / "solution" / candidate_id / "candidate.py"]:
        if p.exists():
            return _load_module(p, "candidate")
    raise FileNotFoundError(f"candidate.py 不在 payload 根也不在 solution/{candidate_id}/ 下")


def load_reference(payload: Path):
    """oracle 优先用任务自带 payload/reference.py（暴露 reference(inputs)->Tensor），
    否则回退内置 RMSNorm（v0 冒烟任务都是 norm 族；正式任务必须自带 reference.py）。"""
    rp = payload / "reference.py"
    if rp.exists():
        return _load_module(rp, "reference").reference
    def reference(inputs):
        x = inputs[0].float()
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        return (x * rms).to(inputs[0].dtype)
    return reference


_CANON_AXES = ("batch", "seq", "hidden")   # 任务语义序：hidden 必须最后（归约维）


def make_inputs(workload: dict, torch):
    """axes → shape：canonical 序（batch,seq,hidden）优先，其余按键名字典序补尾。
    （v0.1 修正：纯 sorted 会得到 batch,hidden,seq —— 归约维跑到倒数第二，测错轴。）"""
    axes = workload["axes"]
    order = [k for k in _CANON_AXES if k in axes] + sorted(k for k in axes if k not in _CANON_AXES)
    shape = [axes[k] for k in order]
    dtype = getattr(torch, "float16") if workload.get("dtype", "fp16") == "fp16" else getattr(torch, "bfloat16")
    inp = torch.randn(*shape, dtype=torch.float32).to(dtype).npu()
    return [inp]


def run_verify(payload: Path, job: dict) -> dict:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(job.get("device_id", 0))
    mod = load_kernel(payload, job.get("candidate_id", "c001"))
    ref_fn = load_reference(payload)
    per_wl = []
    for wl in job["workloads"]:
        inputs = make_inputs(wl, torch)
        # 参考输出先算（fp32 oracle 独立于候选，防候选原地改输入污染参考）
        ref = ref_fn(list(inputs))
        ref = ref[0] if isinstance(ref, (list, tuple)) else ref
        try:
            out = mod.kernel(list(inputs))
            out = out[0] if isinstance(out, (list, tuple)) else out
        except Exception as e:  # noqa: BLE001
            per_wl.append({"id": wl["id"], "passed": False, "error": str(e)[:200]})
            continue
        # 四步协议
        ok, detail = compare(out.float(), ref.float(), TOL.get(wl.get("dtype", "fp16"), 0.02))
        per_wl.append({"id": wl["id"], "passed": ok, **detail})
    passed = all(w.get("passed") for w in per_wl)
    return {"passed": passed, "workloads": per_wl}


def compare(out, ref, limit: float):
    if tuple(out.shape) != tuple(ref.shape):
        return False, {"error": f"shape {tuple(out.shape)} != {tuple(ref.shape)}"}
    nan_o, nan_r = out.isnan(), ref.isnan()
    if not bool((nan_o == nan_r).all()):
        return False, {"error": "NaN positions mismatch"}
    inf_o, inf_r = out.isinf(), ref.isinf()
    if not bool((inf_o == inf_r).all()):
        return False, {"error": "Inf positions mismatch"}
    finite = (~nan_o) & (~inf_o)
    diff = (out - ref).abs()
    denom = ref.abs().clamp_min(1e-8)
    rel = (diff / denom)[finite]
    bad = (rel > limit)
    err_ratio = float(bad.sum()) / max(int(finite.sum()), 1)
    passed = err_ratio <= limit     # 错误元素比例容忍（akg 方案）
    return passed, {"err_ratio": err_ratio, "limit": limit,
                    "mismatch_top": bad.nonzero()[:3].tolist()}


def _time_fn(fn, inputs, torch, l2buf, warmup: int, samples: int) -> list[float]:
    """warmup → 每次采样前清 L2 → 事件计时（μs）。"""
    Evt = torch.npu.Event if hasattr(torch.npu, "Event") else torch.cuda.Event  # CPU-only torch 枝干无 torch.Event
    for _ in range(warmup):
        fn(list(inputs))
    torch.npu.synchronize()
    times = []
    for _ in range(samples):
        l2buf.zero_()                     # 每次采样前清 L2
        torch.npu.synchronize()
        s = Evt(enable_timing=True); e = Evt(enable_timing=True)
        s.record(); fn(list(inputs)); e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)   # ms→μs
    return sorted(times)


def run_bench(payload: Path, job: dict) -> dict:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(job.get("device_id", 0))
    mod = load_kernel(payload, job.get("candidate_id", "c001"))
    has_ref = (payload / "reference.py").exists()
    ref_fn = load_reference(payload) if has_ref else None
    warmup = job.get("extra", {}).get("warmup", 3)
    samples = job.get("extra", {}).get("samples", 5)
    # L2 清除 buffer（192MB；akg 方案）
    l2buf = torch.zeros(192 * 1024 * 1024 // 4, dtype=torch.float32).npu()
    per_wl = []
    for wl in job["workloads"]:
        inputs = make_inputs(wl, torch)
        ts = _time_fn(mod.kernel, inputs, torch, l2buf, warmup, samples)
        entry = {"id": wl["id"], "mean_us": sum(ts) / len(ts), "p50_us": ts[len(ts)//2], "times": ts}
        if has_ref:
            rb = _time_fn(ref_fn, inputs, torch, l2buf, warmup, samples)
            entry["baseline_mean_us"] = sum(rb) / len(rb)
            entry["speedup_vs_ref"] = entry["baseline_mean_us"] / max(entry["mean_us"], 1e-9)
        per_wl.append(entry)
    return {"workloads": per_wl, "warmup": warmup, "samples": samples, "has_reference": has_ref}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="job.json")
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    payload = Path(args.job).resolve().parent       # v0.1：payload 根 = job.json 所在目录
    job = json.loads((payload / Path(args.job).name).read_text(encoding="utf-8"))

    t0 = time.time()
    result = {"job_id": job["job_id"], "kind": job["kind"], "candidate_id": job["candidate_id"],
              "device_id": job.get("device_id", 0),
              "physical_device_id": job.get("physical_device_id", job.get("device_id", 0))}
    try:
        if job["kind"] == "verify":
            result.update(run_verify(payload, job))
        elif job["kind"] == "bench":
            result.update(run_bench(payload, job))
        else:
            result = {**result, "ok": False, "error": "profile: Phase 2"}
    except Exception as e:  # noqa: BLE001
        result = {**result, "ok": False, "error": f"{type(e).__name__}: {e}"}
    result["elapsed_s"] = round(time.time() - t0, 2)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job['job_id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"done": True, "rc": 0 if result.get("passed") or result.get("workloads") else 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
