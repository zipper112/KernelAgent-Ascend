"""tests/test_remote_sync.py —— Job spec 与远程同步的 dry 单测（不实际 ssh）。"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from infra.remote.sync import JobSpec, push_files, _build_exec_cmd  # noqa: E402


def _mk_spec(**kw):
    base = dict(job_id="j001", kind="verify", candidate_id="c001", task="rmsnorm-smoke",
                files=["solution/c001/candidate.py", "reference.py", "bench/workloads.yaml"],
                workloads=[{"id": "w01", "axes": {"batch": 1, "seq": 128, "hidden": 4096}, "dtype": "fp16"}],
                workload_set="l0", device_id=0)
    base.update(kw)
    return JobSpec(**base)


def test_jobspec_roundtrip():
    spec = JobSpec(job_id="j001", kind="verify", candidate_id="c001", task="rmsnorm-v1",
                   files=["solution/c001/candidate.py", "bench/workloads.yaml"],
                   workloads=[{"id": "w01", "axes": {"batch": 1, "seq": 128, "hidden": 4096}, "dtype": "fp16"}],
                   workload_set="l0", device_id=3)
    spec.validate()
    text = spec.to_json()
    back = JobSpec.from_json(text)
    assert back.job_id == "j001" and back.device_id == 3 and back.kind == "verify"
    assert back.workloads[0]["axes"]["hidden"] == 4096


def test_jobspec_rejects_bad_kind():
    with pytest.raises(AssertionError):
        s = JobSpec(job_id="x", kind="deploy", candidate_id="c", task="t", files=[])
        s.validate()


def test_jobspec_rejects_bad_workload_set():
    with pytest.raises(AssertionError):
        s = JobSpec(job_id="x", kind="bench", candidate_id="c", task="t", files=[], workload_set="huge")
        s.validate()


def test_push_missing_files_fails_gracefully(tmp_path):
    """本地缺文件 → 返回错误 dict 而非异常（Executor 契约）。"""
    from infra.remote.executor import RemoteExecutor, RemoteTarget
    ex = RemoteExecutor(RemoteTarget())
    r = ex.push(["no/such/file.py"], local_root=str(tmp_path), remote_dir="/tmp/kda_x")
    assert r["ok"] is False and "本地缺失" in r["error"]


def test_runner_exists_and_selfcontained():
    p = ROOT / "infra" / "remote" / "runner.py"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    for banned in ("from infra", "import infra", "knowledge/", "models.yaml", "GLM"):
        assert banned not in src, f"runner 不许依赖本地件: {banned}"
    for need in ("run_verify", "run_bench", "TOL"):
        assert need in src


def test_container_entry_exists_and_mounts_contract():
    p = ROOT / "infra" / "remote" / "container_entry.sh"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    for need in ("/work/payload", "/work/results", "driver_host", "set_env.sh"):
        assert need in src, f"container_entry 缺挂载/环境约定: {need}"


def test_build_exec_cmd_docker():
    from infra.remote.executor import RemoteTarget
    t = RemoteTarget(exec_mode="docker", docker_image="xllm:glm-clone")
    cmd = _build_exec_cmd(t, _mk_spec(device_id=3), "~/kda-ascend/rmsnorm-smoke/payload",
                          "~/kda-ascend/rmsnorm-smoke/results", "infra/remote/runner.py")
    assert cmd.startswith("sudo docker run --rm")
    assert "--device /dev/davinci3" in cmd          # device_id 直通对应卡
    assert "-v /usr/local/Ascend/driver:/usr/local/Ascend/driver_host:ro" in cmd
    assert "infra/remote/container_entry.sh" in cmd
    assert "/work/payload/infra/remote/runner.py" in cmd
    assert "--results-dir /work/results" in cmd


def test_build_exec_cmd_native_sources_cann_env():
    from infra.remote.executor import RemoteTarget
    t = RemoteTarget(exec_mode="native", cann_env="~/kda-ascend/env.sh")
    cmd = _build_exec_cmd(t, _mk_spec(), "/m/payload", "/m/results", "infra/remote/runner.py")
    assert "source $HOME/kda-ascend/env.sh" in cmd and "docker" not in cmd   # ~/ 须在目标机展开（不被单引号冻结）
    assert "--job /m/payload/job.json" in cmd


def test_build_exec_cmd_docker_requires_image():
    from infra.remote.executor import RemoteTarget
    with pytest.raises(AssertionError):
        _build_exec_cmd(RemoteTarget(exec_mode="docker"), _mk_spec(), "/p", "/r", "infra/remote/runner.py")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
