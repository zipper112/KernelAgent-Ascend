"""tests/test_p1_evidence_gates.py —— P1 批单元测试（双档/链式门/参考可信性/趋势/选课）。

P1-1 workload 双档过滤（dev 集 task.yaml 声明；l0=dev/l1+full=全量）
P1-2 链式终态门（N 连步重放，逐步输出+终态比对；CPU torch 可跑）
P1-3 参考可信性（baseline 3 次自一致；原地演进 reference 不误判）
P1-4 趋势摘要（近 N 轮 direction×mean×verify）
P1-5 对齐轮（round%5==0 review prompt 含停滞检测授权）
P1-6 sha 溯源（job-spec audit 记 reference/workloads 指纹）
P1-7 BitLesson 选课（win 优先+cap）
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.core.evidence import Evidence  # noqa: E402
from harness.core.state import TaskState  # noqa: E402


def _mk_task(tmp_path: Path, name: str = "t") -> Path:
    t = tmp_path / name
    (t / "bench").mkdir(parents=True)
    (t / "solution").mkdir()
    (t / "run").mkdir()
    (t / "task.yaml").write_text(
        "contract:\n  op_family: conv\n  research:\n    symptoms: [latency]\n"
        "  verify_mode: chained\n  chain_steps: 3\n"
        "  workload_sets:\n    dev: [w02]\n    full: [w01, w02, w03]\n", encoding="utf-8")
    (t / "config.yaml").write_text(
        "arch: dav_2201\nbudget: {}\nexecution:\n  remote:\n    enabled: false\n", encoding="utf-8")
    (t / "reference.py").write_text("def reference(i): return i[0]\n", encoding="utf-8")
    (t / "bench" / "workloads.yaml").write_text(
        "workloads:\n"
        "  - id: w01\n    axes: {batch: 1, seq: 8, hidden: 4}\n"
        "  - id: w02\n    axes: {batch: 16, seq: 1, hidden: 4096}\n"
        "  - id: w03\n    axes: {batch: 16, seq: 4, hidden: 4096}\n", encoding="utf-8")
    TaskState.init_for_new_task(t, name)
    return t


# ---------- P1-1 双档 ----------

def test_workload_filter_dev_and_full(tmp_path):
    """给定 task.yaml 声明 dev=[w02] → 则 l0 只跑 w02；l1/full 跑全部 3 行。"""
    import harness.cli as cli
    import yaml
    t = _mk_task(tmp_path)
    wls = yaml.safe_load((t / "bench" / "workloads.yaml")
                         .read_text(encoding="utf-8"))["workloads"]
    dev = cli._filter_workloads(t, wls, "l0")
    assert [w["id"] for w in dev] == ["w02"]
    full = cli._filter_workloads(t, wls, "full")
    assert len(full) == 3


def test_workload_filter_defaults_without_declaration(tmp_path):
    """给定无 workload_sets 声明 → 则 l0 退化为前 2 行（快速档），full 不减。"""
    import harness.cli as cli
    import yaml
    t = _mk_task(tmp_path)
    (t / "task.yaml").write_text("contract:\n  op_family: conv\n", encoding="utf-8")
    wls = yaml.safe_load((t / "bench" / "workloads.yaml")
                         .read_text(encoding="utf-8"))["workloads"]
    assert len(cli._filter_workloads(t, wls, "l0")) == 2
    assert len(cli._filter_workloads(t, wls, "l1")) == 3


# ---------- P1-2 链式终态门 ----------

def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_chained_verify_detects_state_drift(tmp_path):
    """给定候选单步输出正确但 state 更新错（每步漂移累积）→ 链式门在终态或逐步
    比对中抓到（单步协议看不到）。"""
    torch = _torch()
    if torch is None:
        pytest.skip("no torch")
    from infra.remote import runner
    import types
    # reference：y = x + 1（不动 state）；候选：单步输出=y 正确，但每步偷偷
    # 把 inputs[1]（state）乘 1.5——单步输出比对永远过，state 终值发散
    ref = types.ModuleType("ref")
    ref.reference = lambda inputs: inputs[0] + 1
    cand = types.ModuleType("cand")
    def kernel(inputs):
        inputs[1].mul_(1.5)          # state 污染（单步不可见）
        return inputs[0] + 1
    cand.kernel = kernel
    ins = [torch.randn(2, 3), torch.randn(2, 3)]
    r = runner._verify_chained(cand, ref.reference, ins, "w01", 0.02, torch, steps=3)
    assert r["passed"] is False and r.get("gate") == "final-state"


def test_chained_verify_passes_clean_kernel(tmp_path):
    """给定干净候选（输出与 state 都对）→ 则链式门过并标注 state_inputs。"""
    torch = _torch()
    if torch is None:
        pytest.skip("no torch")
    from infra.remote import runner
    import types
    ref = types.ModuleType("ref")
    def reference(inputs):
        inputs[1].add_(1.0)          # reference 也原地演进 state
        return inputs[0] + inputs[1]
    cand = types.ModuleType("cand")
    def kernel(inputs):
        inputs[1].add_(1.0)
        return inputs[0] + inputs[1]
    cand.kernel = kernel
    ins = [torch.randn(2, 3), torch.randn(2, 3)]
    r = runner._verify_chained(cand, reference, ins, "w01", 0.02, torch, steps=3)
    assert r["passed"] is True and 1 in r.get("state_inputs", [])


# ---------- P1-3 参考可信性 ----------

def test_reference_probe_uses_cloned_inputs(tmp_path):
    """给定原地演进 reference → 则 3 次探针从克隆起跑互不污染（bench 语义验证：
    代码路径直接跑 run_bench 的探针段）。"""
    torch = _torch()
    if torch is None:
        pytest.skip("no torch")
    from infra.remote import runner
    import types
    ref = types.ModuleType("ref")
    def reference(inputs):
        inputs[1].add_(1.0)
        return inputs[0]
    ref.reference = reference
    inputs = [torch.randn(2, 2), torch.zeros(2, 2)]
    probes = [ref.reference([t.clone() for t in inputs]) for _ in range(3)]
    stable = all(runner.compare(probes[0].float(), p.float(), 1e-3)[0] for p in probes[1:])
    assert stable                     # 克隆起跑 → 输出稳定（state 演进不背锅）
    assert float(inputs[1].abs().sum()) == 0.0   # 原输入未被探针污染


# ---------- P1-4 趋势摘要 ----------

def test_trend_digest(tmp_path):
    """给定 3 轮档案 → 则趋势表逐轮一行（direction/mean/verify）。"""
    from harness.control.memory import IterationMemory
    m = IterationMemory(_mk_task(tmp_path))
    for r, mean in ((1, 1094.0), (2, 980.0), (3, 606.0)):
        m.save_round(r, {"direction": f"d{r}", "bench": {"mean_us": mean},
                         "verify": {"passed": True}})
    txt = m.trend_digest(3)
    assert "round 1" in txt and "round 3" in txt and "606us" in txt and "过" in txt


# ---------- P1-5 对齐轮 ----------

def test_alignment_round_review(tmp_path, monkeypatch):
    """给定 round=5 的 review → 则 prompt 含全量对齐审计段与停滞检测授权；
    round=4 不含。"""
    from harness.control.loop import AutonomousLoop
    t = _mk_task(tmp_path)
    ev = Evidence(t)
    loop = AutonomousLoop.__new__(AutonomousLoop)
    loop.task, loop.ev, loop.st = t, ev, TaskState(t, ev)
    from harness.control.memory import IterationMemory
    loop.mem = IterationMemory(t)
    loop.models = type("M", (), {"_defaults": {"context_window": 250000},
                                 "chat": staticmethod(lambda *a, **k: "REVISE")})()
    loop._last_review_text = ""
    captured = {}
    def fake_chat(role, messages, **kw):
        captured["p"] = messages[0]["content"]
        return "REVISE"
    loop.models.chat = fake_chat
    loop.review(5, "c005", {"passed": True, "workloads": []},
                {"mean_us": 900.0, "valid": True})
    assert "## 全量对齐审计（本轮 round 5" in captured["p"]     # 渲染段（模板正文里的 SECTION 标记不算）
    loop.review(4, "c004", {"passed": True, "workloads": []},
                {"mean_us": 900.0, "valid": True})
    assert "## 全量对齐审计（本轮 round 4" not in captured["p"]


# ---------- P1-6 sha 溯源 ----------

def test_job_spec_audit_records_shas(tmp_path, monkeypatch):
    """给定 _run_remote_job（ADR-013 版，subprocess/tempfile 全桩）→ 则 audit 的
    job-spec-* 事件带 sha256[:8]；chained 与 dev 档从 job.json 侧证。"""
    import subprocess as _sp
    import tempfile as _tf
    import harness.cli as cli_mod
    t = _mk_task(tmp_path)
    (t / "config.yaml").write_text(
        "execution:\n  remote:\n    enabled: true\n    host: h\n"
        "    exec_mode: docker\n    docker_image: i\n    device_id: 0\n", encoding="utf-8")
    jobs = {}

    class R:
        def __init__(self, rc=0, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(cmd, **kw):
        s = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        if "tar cf -" in s:
            jobs["files"] = s.split("tar cf - ")[1].split(" |")[0].split()
        # 消费并关闭 stdin（Windows 句柄锁——真实 subprocess 会做）
        f = kw.get("stdin")
        if f and hasattr(f, "read"):
            f.read()
            f.close()
        return R(0)
    monkeypatch.setattr(_sp, "run", fake_run)

    _orig_ntf = _tf.NamedTemporaryFile

    class _TF:
        name = "jobstub.json"
        def __init__(self, *a, **k):
            k.pop("delete", None)
            self._f = _orig_ntf(*a, delete=False, **k)
            _TF.name = self._f.name
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self._f.flush()
            import json as _j
            jobs["job"] = _j.loads(open(_TF.name, encoding="utf-8").read())
        def write(self, s):
            self._f.write(s)
    monkeypatch.setattr(_tf, "NamedTemporaryFile", _TF)

    st = TaskState(t, Evidence(t))
    state = st.require()
    cli_mod._run_remote_job(t, state, "verify", "c001", "l0", Evidence(t))
    audit = (t / "docs" / "audit.log").read_text(encoding="utf-8")
    assert "job-spec-verify" in audit and "shas" in audit
    rec = json.loads([l for l in audit.splitlines() if "job-spec-verify" in l][-1])
    assert len(rec["detail"]["shas"]["reference.py"]) == 8
    # 链式任务透传（job.json 侧证）
    assert jobs["job"]["extra"].get("verify_mode") == "chained"
    assert jobs["job"]["extra"].get("chain_steps") == 3
    # dev 档过滤 + payload 同步清单含冻结面
    assert [w["id"] for w in jobs["job"]["workloads"]] == ["w02"]
    assert "reference.py" in jobs["files"] and "bench/workloads.yaml" in jobs["files"]


# ---------- P1-7 选课 ----------

def test_lessons_ranking(tmp_path):
    """给定混合 lessons → 则 win 优先且 cap 生效。"""
    from harness.control.memory import IterationMemory
    m = IterationMemory(_mk_task(tmp_path))
    m.lessons = [
        {"round": 1, "kind": "fail", "text": "f1"},
        {"round": 2, "kind": "win", "text": "w2"},
        {"round": 3, "kind": "fail", "text": "f3"},
        {"round": 4, "kind": "insight", "text": "i4"},
    ]
    txt = m.lessons_digest(n=2)
    # 排序键 (win 优先, round 降序)：w2(win,r2) vs f3(fail) vs i4(insight,r4) vs f1
    # —— win 恒优先；同为非 win 时 round 大者优先 → w2 + i4
    assert "w2" in txt and "i4" in txt and "f1" not in txt and "f3" not in txt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
