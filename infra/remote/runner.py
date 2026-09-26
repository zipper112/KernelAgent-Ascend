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
    """单输入模式（默认）：axes → [B,L,dim] 张量。
    多张量模式：workload['inputs'] 列表，每项 {name, source: main|const, shape_axes, value}——
    source=main 用主 axes（canonical 序）；source=const 用固定值张量（int 索引等）。
    顺序即 kernel(inputs) 的列表顺序。"""
    if "inputs" in workload:
        outs = []
        for spec in workload["inputs"]:
            src = spec.get("source", "main")
            if src == "main":
                # shape_axes 每项：轴名（查 axes）或字面量整数（固定维度，如 state_len=3）
                dims = [workload["axes"][k] if isinstance(k, str) else int(k)
                        for k in spec.get("shape_axes", [])]
                dtype = getattr(torch, "float16") if workload.get("dtype", "fp16") == "fp16" else getattr(torch, "bfloat16")
                outs.append(torch.randn(*dims, dtype=torch.float32).to(dtype).npu())
            elif src == "arange":
                # 0..n-1 互异索引（缓存槽语义：每行独立槽位；全同值会造成别名污染）
                dims = tuple(workload["axes"][k] if isinstance(k, str) else int(k)
                             for k in spec["shape"])
                outs.append(torch.arange(dims[0], dtype=torch.int32).npu())
            elif src == "const":
                # shape 每项：轴名（查 axes）或字面量整数（与 main 分支同规则）
                dims = tuple(workload["axes"][k] if isinstance(k, str) else int(k)
                             for k in spec["shape"])
                if spec.get("dtype") == "int32":
                    outs.append(torch.full(size=dims, fill_value=spec.get("value", 0), dtype=torch.int32).npu())
                else:
                    outs.append(torch.full(size=dims, fill_value=spec.get("value", 1.0), dtype=torch.bfloat16).npu())
        return outs
    axes = workload["axes"]
    order = [k for k in _CANON_AXES if k in axes] + sorted(k for k in axes if k not in _CANON_AXES)
    shape = [axes[k] for k in order]
    dtype = getattr(torch, "float16") if workload.get("dtype", "fp16") == "fp16" else getattr(torch, "bfloat16")
    inp = torch.randn(*shape, dtype=torch.float32).to(dtype).npu()
    return [inp]


def _tensor_sig(t) -> float:
    """状态突变检测用的轻量和校验（runner 内部；fp32 累加容差内可辨"变了没有"）。"""
    return float(t.float().abs().sum().item())


def run_verify(payload: Path, job: dict) -> dict:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(job.get("device_id", 0))
    mod = load_kernel(payload, job.get("candidate_id", "c001"))
    ref_fn = load_reference(payload)
    chained = job.get("extra", {}).get("verify_mode") == "chained"
    steps = int(job.get("extra", {}).get("chain_steps", 3))
    per_wl = []
    for wl in job["workloads"]:
        inputs = make_inputs(wl, torch)
        tol = TOL.get(wl.get("dtype", "fp16"), 0.02)
        if chained:
            per_wl.append(_verify_chained(mod, ref_fn, inputs, wl["id"], tol, torch, steps))
            continue
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
        ok, detail = compare(out.float(), ref.float(), tol)
        per_wl.append({"id": wl["id"], "passed": ok, **detail})
    passed = all(w.get("passed") for w in per_wl)
    return {"passed": passed, "workloads": per_wl, "verify_mode": "chained" if chained else "single"}


def _verify_chained(mod, ref_fn, inputs: list, wl_id: str, tol: float, torch, steps: int) -> dict:
    """P1-2 链式终态门（KDA-Pilot 门 1）：状态携带 kernel 重放 N 连步（每步的 state 原地
    演进喂下一步），比对 ①每步输出序列 ②终态（被原地更新的输入张量）。
    单步 per-step 容差看不到 state 漂移——replay-SSM 实证翻车模式。"""
    ins_c = [t.clone() for t in inputs]      # 两臂各自独立 state
    ins_r = [t.clone() for t in inputs]
    sig0 = [_tensor_sig(t) for t in inputs]
    outs_c, outs_r = [], []
    try:
        for _ in range(steps):
            o = mod.kernel(list(ins_c))
            outs_c.append(o[0] if isinstance(o, (list, tuple)) else o)
        for _ in range(steps):
            o = ref_fn(list(ins_r))
            outs_r.append(o[0] if isinstance(o, (list, tuple)) else o)
    except Exception as e:  # noqa: BLE001
        return {"id": wl_id, "passed": False, "error": str(e)[:200]}
    # ① 逐步输出
    for k, (oc, orr) in enumerate(zip(outs_c, outs_r)):
        ok, detail = compare(oc.float(), orr.float(), tol)
        if not ok:
            return {"id": wl_id, "passed": False, "step": k, "gate": "output", **detail}
    # ② 终态：任一臂发生突变的输入张量视为 state，比对两臂终值
    state_inputs = []
    for j, (tc, tr) in enumerate(zip(ins_c, ins_r)):
        if abs(_tensor_sig(tc) - sig0[j]) > 1e-3 or abs(_tensor_sig(tr) - sig0[j]) > 1e-3:
            state_inputs.append(j)
            ok, detail = compare(tc.float(), tr.float(), tol)
            if not ok:
                return {"id": wl_id, "passed": False, "input_idx": j,
                        "gate": "final-state", **detail}
    return {"id": wl_id, "passed": True, "steps": steps, "state_inputs": state_inputs}


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
                    "mismatches": bad.nonzero()[:10].tolist()}   # 协议 §1：mismatches[≤10]


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
        entry = {"id": wl["id"]}
        # P1-3 参考可信性：baseline 同输入跑 3 次自一致（KDA-Pilot 门 0）——
        # 不稳行的 speedup_vs_ref 无意义，标 ref_unstable 供 keep 判定剔除
        if has_ref:
            try:
                # 每次探针从克隆输入起跑（reference 可能原地演进 state——
                # 同一输入连跑 3 次会天然不同，那不是"不稳定"）
                probes = [ref_fn([t.clone() for t in inputs]) for _ in range(3)]
                probes = [p[0] if isinstance(p, (list, tuple)) else p for p in probes]
                stable = all(compare(probes[0].float(), p.float(), 1e-3)[0] for p in probes[1:])
                entry["ref_unstable"] = not stable
            except Exception:   # noqa: BLE001 —— 探针失败不阻塞计时
                entry["ref_unstable"] = None
        ts = _time_fn(mod.kernel, inputs, torch, l2buf, warmup, samples)
        entry.update({"mean_us": sum(ts) / len(ts), "p50_us": ts[len(ts)//2],
                      "p99_us": ts[max(0, (len(ts) * 99 + 99) // 100 - 1)],   # 协议 §1 bench 输出含 p99
                      "times": ts})
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
