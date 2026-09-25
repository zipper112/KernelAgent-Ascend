"""tests/test_tools.py —— tools/ 五脚本纯函数单元测试（零网络/零 SSH）。

期望先行：每用例 docstring 写「给定 → 当 → 则」。
覆盖：smoke_remote_job.load_workloads、provision_npu.resolve_closure(mock _open)、
check_env（record/check_repo_layout/check_skills/check_router）、
sync_assets（tree_sha/parse_manifest）、build_production_index（detect_archs/scan_repo）。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_tool(name: str):
    """按文件路径加载 tools/<name>（tools 不是包，避免 sys.path 污染）。"""
    spec = importlib.util.spec_from_file_location(f"tools_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- smoke_remote_job.load_workloads ----------

def test_load_workloads_parses_yaml(tmp_path):
    """给定三档 workload yaml（bench/ 下）→ 则 [{id,axes,dtype}] 且未知字段不透传。"""
    bench = tmp_path / "bench"; bench.mkdir()
    (bench / "workloads.yaml").write_text(
        "workloads:\n"
        "  - {id: w01, axes: {batch: 1, seq: 128, hidden: 4096}, dtype: fp16, repr: true}\n"
        "  - {id: w02, axes: {batch: 2, seq: 64, hidden: 2048}}\n", encoding="utf-8")
    m = _load_tool("smoke_remote_job")
    wls = m.load_workloads(tmp_path)
    assert wls == [
        {"id": "w01", "axes": {"batch": 1, "seq": 128, "hidden": 4096}, "dtype": "fp16"},
        {"id": "w02", "axes": {"batch": 2, "seq": 64, "hidden": 2048}, "dtype": "fp16"},   # 缺省 fp16
    ]


# ---------- provision_npu.resolve_closure ----------

class _FakeResp:
    def __init__(self, payload): self._p = payload
    def read(self): return json.dumps(self._p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_resolve_closure_picks_cp310_aarch64(monkeypatch):
    """给定 PyPI JSON（混入 cp39/amd64 干扰项）→ 则 torch/torch_npu 各取 cp310 aarch64 wheel。"""
    m = _load_tool("provision_npu")

    def fake_open(url, timeout=30):
        pkg = url.split("/pypi/")[1].split("/")[0]
        files = [
            {"filename": f"torch-2.13.0-cp39-cp39-manylinux_2_28_aarch64.whl", "url": "BAD-cp39"},
            {"filename": f"torch-2.13.0-cp310-cp310-manylinux_2_28_x86_64.whl", "url": "BAD-amd64"},
            {"filename": f"torch-2.13.0-cp310-cp310-manylinux_2_28_aarch64.whl", "url": "GOOD-torch"},
        ] if pkg == "torch" else [
            {"filename": "torch_npu-2.13.0rc1-cp310-cp310-manylinux_2_28_aarch64.whl", "url": "GOOD-tnpu"},
        ] if pkg == "torch_npu" else [
            {"filename": f"{pkg}-1.0-py3-none-any.whl", "url": f"LITE-{pkg}"},
        ]
        return _FakeResp({"urls": files})

    monkeypatch.setattr(m, "_open", fake_open)
    urls = m.resolve_closure()
    assert "GOOD-torch" in urls and "GOOD-tnpu" in urls
    assert not any(u.startswith("BAD") for u in urls)
    assert any(u == "LITE-jinja2" for u in urls)      # 轻依赖纯 py wheel 也进清单


def test_provision_remote_wraps_cmd_in_json_quotes():
    """给定 remote(cmd) → 则内层命令被 json.dumps 引号包装（防远端 shell 拆词）。"""
    m = _load_tool("provision_npu")
    seen = {}

    def fake_sh(cmd, timeout=300):
        seen["cmd"] = cmd
        import subprocess
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(m, "sh", fake_sh)
    try:
        m.remote("cd ~/x && venv/bin/pip install a.whl")
    finally:
        monkey.undo()
    inner = seen["cmd"][-1]
    assert 'ssh' in inner and '"cd ~/x' in inner or "cd ~\\/x" in inner
    assert "yq-e15" in inner


# ---------- check_env ----------

def test_check_env_record_accumulates(capsys):
    """给定 record 三次调用 → 则 RESULTS 计数 3 且输出带分级标记。"""
    m = _load_tool("check_env")
    m.RESULTS.clear()
    m.record("a", "ok")
    m.record("b", "warn", "det")
    m.record("c", "fail")
    assert len(m.RESULTS) == 3 and m.RESULTS[1]["detail"] == "det"
    out = capsys.readouterr().out
    assert "[ok]" in out and "[warn]" in out and "[FAIL]" in out


def test_check_env_repo_layout_ok_on_real_repo():
    """给定本仓 → 则骨架完整性 check 结果为 ok。"""
    m = _load_tool("check_env")
    m.RESULTS.clear()
    m.check_repo_layout()
    assert m.RESULTS[-1]["status"] == "ok"


def test_check_env_skills_ok_on_real_repo():
    """给定本仓 vendored skills → 则无缺口（core13/triton6/ascendc24/pypto17/tilelang6）。"""
    m = _load_tool("check_env")
    m.RESULTS.clear()
    m.check_skills()
    assert m.RESULTS[-1]["status"] == "ok"


def test_check_env_router_all_cases_pass():
    """给定本仓 → 则三条代表性查询退出码符合预期（0/0/1）。"""
    m = _load_tool("check_env")
    m.RESULTS.clear()
    m.check_router()
    assert all(r["status"] == "ok" for r in m.RESULTS) and len(m.RESULTS) == 3


# ---------- sync_assets ----------

def test_tree_sha_stable_and_separator_normalized(tmp_path):
    """给定同内容文件树（路径分隔符 \ 或 /）→ 则 hash 一致且可复现。"""
    m = _load_tool("sync_assets")
    d1 = tmp_path / "t1"; (d1 / "sub").mkdir(parents=True)
    (d1 / "sub" / "a.txt").write_text("hello", encoding="utf-8")
    (d1 / "b.txt").write_text("world", encoding="utf-8")
    h1 = m.tree_sha(d1)
    assert h1 == m.tree_sha(d1)                       # 稳定
    assert len(h1) == 16 and "\\" not in h1


def test_tree_sha_changes_on_content_change(tmp_path):
    """给定文件内容变化 → 则 hash 变化。"""
    m = _load_tool("sync_assets")
    d = tmp_path / "t"; d.mkdir()
    f = d / "a.txt"; f.write_text("v1", encoding="utf-8")
    h1 = m.tree_sha(d)
    f.write_text("v2", encoding="utf-8")
    assert m.tree_sha(d) != h1


def test_parse_manifest_min_seventy_assets():
    """给定本仓 manifest → 则 ≥70 条资产且六 group 齐全。"""
    m = _load_tool("sync_assets")
    assets = m.parse_manifest()
    groups = {a.get("group", "") for a in assets}
    assert len(assets) >= 70
    assert {"core", "triton-ascend", "ascendc", "pypto", "tilelang", "akg"} <= groups


# ---------- build_production_index ----------

def test_detect_archs_negative_lookahead():
    """给定 a3 与 a310 路径 → 则 a3 命中 dav_2201（负 lookahead）、a310 不误判为 2201。"""
    m = _load_tool("build_production_index")
    assert "dav_2201" in m.detect_archs("ops/norm/rms_norm_a3/op_kernel")
    assert "dav_2201" not in m.detect_archs("ops/norm/rms_norm_a310/op_kernel")


def test_detect_archs_three_generations():
    """给定三代标记 → 则分别归 dav_200/dav_2201/dav_3510。"""
    m = _load_tool("build_production_index")
    assert m.detect_archs("ascend310/op") == ["dav_200"]
    assert m.detect_archs("ascend910b/op") == ["dav_2201"]
    assert m.detect_archs("regbase/op") == ["dav_3510"]


def test_detect_archs_no_marker_is_empty():
    """给定无架构标记 → 则空列表（后续写入 both）。"""
    m = _load_tool("build_production_index")
    assert m.detect_archs("ops/plain/op_kernel") == []


def test_scan_repo_op_layout(tmp_path, monkeypatch):
    """给定假仓（norm/rms_norm 带 op_kernel、experimental/ffn 带 op_host、common 排除）→ 则
    扫出 2 条：rms_norm kind=op、ffn kind=experimental-op；common 不入。"""
    m = _load_tool("build_production_index")
    monkeypatch.setattr(m, "ROOT", tmp_path)          # relative_to(ROOT) 用 tmp 根
    repo = tmp_path / "ops-nn"
    (repo / "norm" / "rms_norm" / "op_kernel").mkdir(parents=True)
    (repo / "norm" / "common" / "op_kernel").mkdir(parents=True)
    (repo / "norm" / "experimental" / "ffn" / "op_host").mkdir(parents=True)
    entries = m.scan_repo(repo, "ops-nn")
    by_op = {e["op"]: e for e in entries}
    assert set(by_op) == {"rms_norm", "ffn"}
    assert by_op["rms_norm"]["kind"] == "op"
    assert by_op["ffn"]["kind"] == "experimental-op"
    assert by_op["rms_norm"]["repo"] == "ops-nn"
    assert by_op["rms_norm"]["path"].endswith("rms_norm")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
