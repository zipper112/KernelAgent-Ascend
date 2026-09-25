"""tests/test_runner_offline.py —— infra/remote/runner.py 全离线单元测试（CPU torch + NPU 桩）。

期望先行：每个用例 docstring 写明「给定 → 当 → 则」。
桩策略：monkeypatch torch.Tensor.npu=identity、torch.npu=SimpleNamespace、sys.modules['torch_npu']=空模块，
不依赖任何 SSH/NPU/网络。
"""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.remote import runner  # noqa: E402

torch = pytest.importorskip("torch", reason="runner 张量逻辑需 CPU torch")


# ---------- NPU 桩 ----------

class FakeEvent:
    created = 0

    def __init__(self, enable_timing=False):
        self.seq = FakeEvent.created
        FakeEvent.created += 1

    def record(self):
        pass

    def elapsed_time(self, other):
        return 2.0   # 固定 2ms → 2000μs


@pytest.fixture
def npu_env(monkeypatch):
    """给定 CPU torch → 注入 NPU 桩 → runner 全部张量路径可离线跑。"""
    calls = {"set_device": []}
    stub = types.ModuleType("torch_npu")
    monkeypatch.setitem(sys.modules, "torch_npu", stub)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        set_device=lambda i: calls["set_device"].append(i),
        synchronize=lambda: None,
        Event=FakeEvent), raising=False)
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    return calls


# ---------- compare 四步协议 ----------

def test_compare_shape_mismatch_fails():
    """给定 out/ref 形状不同 → 当 compare → 则 (False, error 含 shape)。"""
    ok, d = runner.compare(torch.zeros(2, 3), torch.zeros(3, 2), 0.004)
    assert ok is False and "shape" in d["error"]


def test_compare_nan_position_mismatch_fails():
    """给定 NaN 位置不同的同形张量 → 则 False 且 error=NaN positions mismatch。"""
    out = torch.zeros(4); out[1] = float("nan")
    ref = torch.zeros(4); ref[2] = float("nan")
    ok, d = runner.compare(out, ref, 0.004)
    assert ok is False and "NaN" in d["error"]


def test_compare_inf_position_mismatch_fails():
    """给定 Inf 位置不同 → 则 False 且 error=Inf。"""
    out = torch.zeros(4); out[0] = float("inf")
    ref = torch.zeros(4); ref[3] = float("inf")
    ok, d = runner.compare(out, ref, 0.004)
    assert ok is False and "Inf" in d["error"]


def test_compare_small_error_passes():
    """给定全对齐且误差远小于 limit → 则 True 且 err_ratio≈0。"""
    ref = torch.ones(100)
    out = ref + 1e-6
    ok, d = runner.compare(out, ref, 0.004)
    assert ok is True and d["err_ratio"] == 0.0


def test_compare_err_ratio_at_limit_passes():
    """给定坏元素比例恰好等于 limit → 则通过（err_ratio<=limit 的 <= 语义）。"""
    ref = torch.ones(100)
    out = ref.clone()
    out[:1] = 100.0   # 1/100 = 0.01 == limit
    ok, d = runner.compare(out, ref, 0.01)
    assert ok is True and d["err_ratio"] == pytest.approx(0.01)


def test_compare_err_ratio_over_limit_fails_with_coords():
    """给定坏元素比例超 limit → 则 False 且 mismatches 给出坐标（协议 §1 ≤10 条）。"""
    ref = torch.ones(100)
    out = ref.clone(); out[:5] = 100.0
    ok, d = runner.compare(out, ref, 0.01)
    assert ok is False and 0.0 < d["err_ratio"] <= 0.06 and 1 <= len(d["mismatches"]) <= 10


def test_compare_aligned_nans_pass():
    """给定 NaN 同位（finite 掩码正确排除）→ 则 True。"""
    ref = torch.ones(8); ref[3] = float("nan")
    out = ref.clone()
    ok, d = runner.compare(out, ref, 0.004)
    assert ok is True


# ---------- 候选与 oracle 加载 ----------

def test_load_kernel_flat_layout(tmp_path):
    """给定 payload/candidate.py（扁平布局）→ 则加载成功。"""
    (tmp_path / "candidate.py").write_text("def kernel(inputs): return inputs[0]\n", encoding="utf-8")
    mod = runner.load_kernel(tmp_path, "c001")
    assert callable(mod.kernel)


def test_load_kernel_solution_layout(tmp_path):
    """给定 payload/solution/c001/candidate.py（仓内布局）→ 则加载成功。"""
    d = tmp_path / "solution" / "c001"
    d.mkdir(parents=True)
    (d / "candidate.py").write_text("def kernel(inputs): return inputs[0]\n", encoding="utf-8")
    assert callable(runner.load_kernel(tmp_path, "c001").kernel)


def test_load_kernel_missing_raises(tmp_path):
    """给定 payload 无 candidate.py → 则 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        runner.load_kernel(tmp_path, "c001")


def test_load_reference_fallback_math(tmp_path):
    """给定 payload 无 reference.py → 则回退内置 RMSNorm，数学上与手写公式一致（max_diff<1e-6）。"""
    ref = runner.load_reference(tmp_path)
    x = torch.randn(4, 128, dtype=torch.float32)
    got = ref([x])
    want = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert float((got - want).abs().max()) < 1e-6


def test_load_reference_prefers_task_file(tmp_path):
    """给定 payload/reference.py → 则用任务自带 oracle（返回可辨识的常值）。"""
    (tmp_path / "reference.py").write_text(
        "def reference(inputs):\n    return inputs[0] * 0.0\n", encoding="utf-8")
    got = runner.load_reference(tmp_path)([torch.ones(3)])
    assert float(got.abs().max()) == 0.0


# ---------- make_inputs ----------

def test_make_inputs_canonical_axis_order(npu_env):
    """给定 axes={seq,batch,hidden} 乱序 → 则 shape 按 canonical (batch,seq,hidden) 展开——
    hidden 必须是最后一维（归约维；v0.1 修复前 sorted 会得到 batch,hidden,seq）。"""
    wl = {"id": "w", "axes": {"seq": 5, "hidden": 7, "batch": 3}, "dtype": "fp16"}
    (inp,) = runner.make_inputs(wl, torch)
    assert tuple(inp.shape) == (3, 5, 7) and inp.dtype == torch.float16


def test_make_inputs_extra_axes_sorted_tail(npu_env):
    """给定含非 canonical 轴 → 则 canonical 轴在前、其余字典序补尾。"""
    wl = {"id": "w", "axes": {"hidden": 4, "zz": 2, "batch": 3, "aa": 1}}
    (inp,) = runner.make_inputs(wl, torch)
    assert tuple(inp.shape) == (3, 4, 1, 2)


def test_make_inputs_bf16_dtype(npu_env):
    """给定 dtype=bf16 → 则映射 torch.bfloat16。"""
    wl = {"id": "w", "axes": {"batch": 1, "seq": 2, "hidden": 4}, "dtype": "bf16"}
    (inp,) = runner.make_inputs(wl, torch)
    assert inp.dtype == torch.bfloat16


# ---------- run_verify / run_bench ----------

GOOD_KERNEL = (
    "import torch\n"
    "def kernel(inputs):\n"
    "    x = inputs[0]\n"
    "    xf = x.float()\n"
    "    ms = xf.pow(2).mean(-1, keepdim=True)\n"
    "    return (xf / torch.sqrt(ms + 1e-6)).to(x.dtype)\n"
)


def _mk_job(kind="verify"):
    return {"job_id": "t1", "kind": kind, "candidate_id": "c001", "device_id": 0,
            "workloads": [{"id": "w01", "axes": {"batch": 2, "seq": 3, "hidden": 16}, "dtype": "fp16"}],
            "extra": {"warmup": 2, "samples": 3}}


def test_run_verify_passes_correct_kernel(tmp_path, npu_env):
    """给定与 oracle 等价的 kernel → 则 passed=True 且调用 set_device(0)。"""
    (tmp_path / "solution" / "c001").mkdir(parents=True)
    (tmp_path / "solution" / "c001" / "candidate.py").write_text(GOOD_KERNEL, encoding="utf-8")
    r = runner.run_verify(tmp_path, _mk_job())
    assert r["passed"] is True and r["workloads"][0]["passed"] is True
    assert npu_env["set_device"] == [0]


def test_run_verify_kernel_exception_recorded(tmp_path, npu_env):
    """给定抛异常的 kernel → 则该 workload 记 error、整体 passed=False、不向上抛。"""
    (tmp_path / "candidate.py").write_text(
        "def kernel(inputs):\n    raise ValueError('boom')\n", encoding="utf-8")
    r = runner.run_verify(tmp_path, _mk_job())
    assert r["passed"] is False and "boom" in r["workloads"][0]["error"]


def test_run_verify_ref_computed_before_kernel(tmp_path, npu_env):
    """给定计算正确但随后原地清零输入的 kernel → 则仍 passed=True（证明 ref 先算，不被候选污染）。"""
    src = GOOD_KERNEL + "    inputs[0].mul_(0.0)\n"
    # mul_ 需在 return 前执行：重写为 destroy-then-return 形式
    src = (
        "import torch\n"
        "def kernel(inputs):\n"
        "    x = inputs[0]\n"
        "    xf = x.float()\n"
        "    ms = xf.pow(2).mean(-1, keepdim=True)\n"
        "    out = (xf / torch.sqrt(ms + 1e-6)).to(x.dtype)\n"
        "    inputs[0].mul_(0.0)\n"
        "    return out\n"
    )
    (tmp_path / "candidate.py").write_text(src, encoding="utf-8")
    r = runner.run_verify(tmp_path, _mk_job())
    assert r["passed"] is True


def test_run_bench_with_reference_reports_speedup(tmp_path, npu_env):
    """给定 payload 带 reference.py → 则每 workload 含 baseline_mean_us 与 speedup_vs_ref，且 p99 字段在（协议 §1）。"""
    (tmp_path / "candidate.py").write_text(GOOD_KERNEL, encoding="utf-8")
    (tmp_path / "reference.py").write_text(
        "def reference(inputs):\n    return inputs[0]\n", encoding="utf-8")
    r = runner.run_bench(tmp_path, _mk_job("bench"))
    w = r["workloads"][0]
    assert r["has_reference"] is True and w["baseline_mean_us"] > 0 and w["speedup_vs_ref"] > 0
    assert r["warmup"] == 2 and r["samples"] == 3
    assert w["p99_us"] >= w["p50_us"] > 0


def test_run_bench_without_reference_no_baseline(tmp_path, npu_env):
    """给定 payload 无 reference.py → 则 has_reference=False 且无 baseline/speedup 键。"""
    (tmp_path / "candidate.py").write_text(GOOD_KERNEL, encoding="utf-8")
    r = runner.run_bench(tmp_path, _mk_job("bench"))
    w = r["workloads"][0]
    assert r["has_reference"] is False
    assert "baseline_mean_us" not in w and "speedup_vs_ref" not in w
    assert w["mean_us"] > 0


# ---------- _time_fn ----------

def test_time_fn_warmup_l2_and_units(npu_env):
    """给定 FakeEvent(2ms) 与计数 kernel/L2 桩 → 则 warmup 恰好 N 次、L2 zero_ 恰好 samples 次、
    返回升序且毫秒→微斯换算（2000.0）。"""
    calls = {"fn": 0}

    class L2:
        zero_calls = 0

        def zero_(self):
            L2.zero_calls += 1

    def fn(inputs):
        calls["fn"] += 1
        return inputs[0]

    times = runner._time_fn(fn, [torch.ones(2)], torch, L2(), warmup=2, samples=3)
    assert calls["fn"] == 2 + 3          # warmup + 采样各一次
    assert L2.zero_calls == 3            # 每次采样前清一次 L2
    assert times == sorted(times) == [2000.0] * 3


# ---------- main ----------

def test_main_profile_not_implemented(tmp_path, monkeypatch, npu_env):
    """给定 kind=profile → 则 result 含 'profile: Phase 2'、退出码 0、JSON 落盘。"""
    job = _mk_job("profile")
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    out = tmp_path / "results"
    monkeypatch.chdir(tmp_path)   # 生产形态：container_entry cd /work/payload 后传相对 job.json
    monkeypatch.setattr(sys, "argv", ["runner.py", "--job", "job.json", "--results-dir", str(out)])
    rc = runner.main()
    data = json.loads((out / "t1.json").read_text(encoding="utf-8"))
    assert rc == 0 and "profile: Phase 2" in data["error"]


def test_main_kernel_error_lands_in_result(tmp_path, monkeypatch, npu_env):
    """给定 verify 但无 candidate.py → 则异常被捕获写入 result JSON 的 error 字段。"""
    (tmp_path / "job.json").write_text(json.dumps(_mk_job()), encoding="utf-8")
    out = tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner.py", "--job", "job.json", "--results-dir", str(out)])
    rc = runner.main()
    data = json.loads((out / "t1.json").read_text(encoding="utf-8"))
    assert rc == 0 and "FileNotFoundError" in data["error"] and data["elapsed_s"] >= 0


def test_main_verify_end_to_end(tmp_path, monkeypatch, npu_env):
    """给定扁平布局正确 kernel → 则 main 全流程 passed=True 落盘。"""
    (tmp_path / "candidate.py").write_text(GOOD_KERNEL, encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps(_mk_job()), encoding="utf-8")
    out = tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner.py", "--job", "job.json", "--results-dir", str(out)])
    assert runner.main() == 0
    data = json.loads((out / "t1.json").read_text(encoding="utf-8"))
    assert data["passed"] is True and data["physical_device_id"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
