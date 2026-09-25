"""tests/test_sync_transport.py —— infra/remote/sync.py 传输层单元测试（mock subprocess，零 SSH）。

期望先行：每用例 docstring 写「给定 → 当 → 则」。
桩策略：monkeypatch sync.subprocess.run 返回构造的 CompletedProcess，校验调用序列与返回 dict 契约
（失败不抛异常、错误进 dict、stage 标注）。
"""
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.remote import sync  # noqa: E402
from infra.remote.executor import RemoteTarget  # noqa: E402

TGT = RemoteTarget(jump="jump", host="yq-e15", exec_mode="docker", docker_image="img:1")


def _cp(rc=0, out=b"", err=b""):
    """生产路径 capture_output 且无 text=True → stdout/stderr 恒 bytes，mock 对齐。"""
    return subprocess.CompletedProcess(args=[], returncode=rc,
                                       stdout=out.encode() if isinstance(out, str) else out,
                                       stderr=err.encode() if isinstance(err, str) else err)


# ---------- _q ----------

def test_q_tilde_prefix_expands():
    """给定 ~/a b → 则 $HOME/'a b'（~ 不被单引号冻结）。"""
    assert sync._q("~/a b") == "$HOME/'a b'"


def test_q_plain_path_quoted():
    """给定普通路径 → 则 shlex.quote 原样。"""
    assert sync._q("/m/x y") == shlex.quote("/m/x y")


def test_q_bare_tilde_expands_home():
    """给定裸 ~ → 则 $HOME（同样不被单引号冻结）。"""
    assert sync._q("~") == "$HOME"


# ---------- JobSpec.validate bench 分支 ----------

def test_jobspec_bench_zero_warmup_rejected():
    """给定 bench + warmup=0 → 则 AssertionError。"""
    s = sync.JobSpec(job_id="j", kind="bench", candidate_id="c", task="t", files=[],
                     extra={"warmup": 0, "samples": 3})
    with pytest.raises(AssertionError):
        s.validate()


def test_jobspec_bench_zero_samples_rejected():
    """给定 bench + samples=0 → 则 AssertionError。"""
    s = sync.JobSpec(job_id="j", kind="bench", candidate_id="c", task="t", files=[],
                     extra={"warmup": 2, "samples": 0})
    with pytest.raises(AssertionError):
        s.validate()


# ---------- push_files ----------

def test_push_files_success_three_legs(tmp_path, monkeypatch):
    """给定两个本地文件 + 三段 ssh 全 rc=0 → 则 ok=True pushed=2，且恰好三段调用：
    ①jump 收 tar（cat >） ②host 收包（mkdir -p） ③host 解包（tar xf）。"""
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("x=1\n", encoding="utf-8")
    legs = []

    def fake_run(cmd, **kw):
        remote = cmd[-1] if isinstance(cmd, list) else str(cmd)
        legs.append(remote)
        return _cp(0)

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.push_files(TGT, tmp_path, ["a.py", "b.py"], "~/kda-ascend/t/payload")
    assert r["ok"] is True and r["pushed"] == 2 and len(legs) == 3
    assert "cat >" in legs[0] and "mkdir -p" in legs[1] and "tar xf" in legs[2]


def test_push_files_jump_failure_returns_dict(tmp_path, monkeypatch):
    """给定 jump 段 rc=1 → 则 {ok:False, error 含 jump 传输失败}，不抛异常。"""
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")

    def fake_run(cmd, **kw):
        remote = cmd[-1] if isinstance(cmd, list) else str(cmd)
        if remote.startswith("cat >"):          # 第一段：jump 收 tar
            return _cp(1, err="ssh: connect fail")
        return _cp(0)

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.push_files(TGT, tmp_path, ["a.py"], "/tmp/p")
    assert r["ok"] is False and "jump" in r["error"]


def test_push_files_host_failure_returns_dict(tmp_path, monkeypatch):
    """给定 host 段 rc=255 → 则 error 含 host 传输失败。"""
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    state = {"n": 0}

    def fake_run(cmd, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return _cp(0)
        return _cp(255, err="host unreachable")

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.push_files(TGT, tmp_path, ["a.py"], "/tmp/p")
    assert r["ok"] is False and "host" in r["error"]


def test_push_files_untar_failure_returns_dict(tmp_path, monkeypatch):
    """给定解包段 rc≠0 → 则 error 含 解包失败。"""
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    state = {"n": 0}

    def fake_run(cmd, **kw):
        state["n"] += 1
        return _cp(0) if state["n"] < 3 else _cp(2, err="tar: corrupt")

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.push_files(TGT, tmp_path, ["a.py"], "/tmp/p")
    assert r["ok"] is False and "解包" in r["error"]


# ---------- pull_files ----------

def _tar_bytes(names_contents: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for n, c in names_contents.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(c)
            tf.addfile(ti, io.BytesIO(c))
    return buf.getvalue()


def test_pull_files_extracts_flat(tmp_path, monkeypatch):
    """给定远端返回含 result.json 的 tar 字节 → 则 ok=True、成员打平落在 local_dir。"""
    payload = _tar_bytes({"result.json": b'{"passed": true}'})

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr=b"")

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    out = tmp_path / "out"
    r = sync.pull_files(TGT, ["~/kda-ascend/t/results/j1.json"], out)
    assert r["ok"] is True and r["pulled"] == 1
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["passed"] is True


def test_pull_files_filters_directories(tmp_path, monkeypatch):
    """给定 tar 含目录成员与文件 → 则只计文件、目录条目被跳过。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        d = tarfile.TarInfo("sub/")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        c = b"data"
        ti = tarfile.TarInfo("sub/f.txt"); ti.size = len(c)
        tf.addfile(ti, io.BytesIO(c))
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=buf.getvalue(), stderr=b"")
    monkeypatch.setattr(sync.subprocess, "run", lambda *a, **k: fake)
    r = sync.pull_files(TGT, ["x"], tmp_path / "o")
    assert r["ok"] is True and r["pulled"] == 1


def test_pull_files_failure_returns_dict(tmp_path, monkeypatch):
    """给定远端 rc=255/空 stdout → 则 {ok:False, error 含 拉取失败}，不抛异常。"""
    fake = subprocess.CompletedProcess(args=[], returncode=255, stdout=b"", stderr=b"down")
    monkeypatch.setattr(sync.subprocess, "run", lambda *a, **k: fake)
    r = sync.pull_files(TGT, ["x"], tmp_path / "o")
    assert r["ok"] is False and "拉取" in r["error"] and r["pulled"] == 0


# ---------- run_job ----------

def _mk_repo(tmp_path):
    """造最小仓形态：repo_root 有 infra/remote/runner.py+entry；task_root 有 solution+reference。"""
    (tmp_path / "infra" / "remote").mkdir(parents=True)
    (tmp_path / "infra" / "remote" / "runner.py").write_text("# runner\n", encoding="utf-8")
    (tmp_path / "infra" / "remote" / "container_entry.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    task = tmp_path / "tasks" / "rmsnorm-smoke"
    (task / "solution" / "c001").mkdir(parents=True)
    (task / "solution" / "c001" / "candidate.py").write_text("def kernel(i): return i[0]\n", encoding="utf-8")
    (task / "reference.py").write_text("def reference(i): return i[0]\n", encoding="utf-8")
    return tmp_path, task


SPEC = dict(job_id="j9", kind="verify", candidate_id="c001", task="rmsnorm-smoke",
            files=["solution/c001/candidate.py", "reference.py"], device_id=7,
            workloads=[{"id": "w01", "axes": {"batch": 1, "seq": 2, "hidden": 4}, "dtype": "fp16"}])


def test_run_job_stages_two_sources_and_job_json(tmp_path, monkeypatch):
    """给定 docker 目标 + 仓/任务两源文件 → 则 push 的 tar 里同时含 runner/container_entry/任务件，
    job.json 内容 device_id=0 且 physical_device_id=7。"""
    repo, task = _mk_repo(tmp_path)
    job_json_seen = {}

    def fake_run(cmd, **kw):
        if kw.get("input") is not None:
            # job.json stdin 通道
            job_json_seen["text"] = kw["input"].decode() if isinstance(kw["input"], bytes) else kw["input"]
            return _cp(0)
        return _cp(0)

    # push 阶段真跑（tar 打包可见），exec/pull mock
    real_run = sync.subprocess.run
    result_tar = {}

    def spy_run(cmd, **kw):
        if kw.get("input") is not None and isinstance(kw.get("input"), bytes) and b"job_id" in kw["input"]:
            job_json_seen["text"] = kw["input"].decode()
            return _cp(0)
        # push 的第一段（jump 收 tar）：捕获 tar 字节
        if kw.get("input") is not None:
            result_tar["bytes"] = kw["input"]
            return _cp(0)
        if "docker run" in str(cmd):
            return _cp(0, out='{"done": true, "rc": 0}\n')
        # pull：返回结果 tar
        rt = _tar_bytes({"j9.json": b'{"passed": true, "job_id": "j9"}'})
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=rt, stderr=b"")

    monkeypatch.setattr(sync.subprocess, "run", spy_run)
    out = tmp_path / "results_out"
    r = sync.run_job(TGT, sync.JobSpec(**SPEC), "~/kda-ascend", "infra/remote/runner.py",
                     task, out, repo_root=repo)
    assert r["ok"] is True and r["stage"] == "done"
    assert r["result"]["passed"] is True
    # job.json 契约：docker 模式 device 归零 + physical 备查
    assert json.loads(job_json_seen["text"])["device_id"] == 0
    assert json.loads(job_json_seen["text"])["physical_device_id"] == 7
    # push tar 里两源文件都在（staging 合流）
    with tarfile.open(fileobj=io.BytesIO(result_tar["bytes"])) as tf:
        names = set(tf.getnames())
    assert {"infra/remote/runner.py", "infra/remote/container_entry.sh",
            "solution/c001/candidate.py", "reference.py"} <= names


def test_run_job_pull_failure_stage_marked(tmp_path, monkeypatch):
    """给定 pull 段失败 → 则 stage='pull' 且 ok=False。"""
    repo, task = _mk_repo(tmp_path)

    def fake_run(cmd, **kw):
        s = str(cmd)
        if kw.get("input") is not None:
            return _cp(0)                       # push 段1 / job.json（stdin 通道）
        if "mkdir -p" in s or "tar xf" in s:
            return _cp(0)                       # push 段2/3
        if "docker run" in s:
            return _cp(0)                       # exec
        return _cp(1, err="gone")               # 剩下的即 pull

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.run_job(TGT, sync.JobSpec(**SPEC), "~/kda-ascend", "infra/remote/runner.py",
                     task, tmp_path / "o", repo_root=repo)
    assert r["ok"] is False and r["stage"] == "pull"


def test_run_job_missing_repo_file_stage_error(tmp_path, monkeypatch):
    """给定 repo_root 缺 runner.py → 则 stage='stage'、error 含本地缺失、不发起任何 ssh。"""
    repo, task = _mk_repo(tmp_path)
    (repo / "infra" / "remote" / "runner.py").unlink()
    called = []
    monkeypatch.setattr(sync.subprocess, "run", lambda *a, **k: called.append(1) or _cp(0))
    r = sync.run_job(TGT, sync.JobSpec(**SPEC), "~/kda-ascend", "infra/remote/runner.py",
                     task, tmp_path / "o", repo_root=repo)
    assert r["ok"] is False and r["stage"] == "stage" and "本地缺失" in r["error"]
    assert not called


def test_run_job_native_uses_cann_env(tmp_path, monkeypatch):
    """给定 native 目标（cann_env）→ 则 exec 命令含 source $HOME/...、无 docker run。"""
    repo, task = _mk_repo(tmp_path)
    seen = {}

    def fake_run(cmd, **kw):
        s = str(cmd)
        if kw.get("input") is not None and isinstance(kw.get("input"), bytes) and b"job_id" in kw["input"]:
            return _cp(0)
        if kw.get("input") is not None:
            return _cp(0)
        if "timeout" in s and "ssh" in s and ("python3" in s or "docker" in s):
            seen["exec"] = s
            return _cp(0)
        rt = _tar_bytes({"j9.json": b"{}"})
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=rt, stderr=b"")

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    tgt = RemoteTarget(exec_mode="native", cann_env="~/kda-ascend/env.sh")
    r = sync.run_job(tgt, sync.JobSpec(**SPEC), "~/kda-ascend", "infra/remote/runner.py",
                     task, tmp_path / "o", repo_root=repo)
    assert r["ok"] is True
    assert "source $HOME/kda-ascend/env.sh" in seen["exec"] and "docker" not in seen["exec"]


def test_run_job_exec_timeout_returns_dict_not_raises(tmp_path, monkeypatch):
    """给定 exec 段链路超时（TimeoutExpired）→ 则返回 {ok:False, stage:'exec'} 且带
    results_path 提示可重试 pull——不抛异常（§8a 契约，B-7 修复）。"""
    repo, task = _mk_repo(tmp_path)

    def fake_run(cmd, **kw):
        s = str(cmd)
        if kw.get("input") is not None:
            return _cp(0)
        if "mkdir -p" in s or "tar xf" in s:
            return _cp(0)
        if "docker run" in s:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=99)
        return _cp(0)

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.run_job(TGT, sync.JobSpec(**SPEC), "~/kda-ascend", "infra/remote/runner.py",
                     task, tmp_path / "o", repo_root=repo)
    assert r["ok"] is False and r["stage"] == "exec"
    assert "link-timeout" in r["error"] and r["results_path"].endswith("j9.json")


def test_run_job_jobjson_timeout_returns_dict(tmp_path, monkeypatch):
    """给定 job.json 段超时 → 则 {ok:False, stage:'job.json'}，不抛异常。"""
    repo, task = _mk_repo(tmp_path)

    def fake_run(cmd, **kw):
        if kw.get("input") is not None and b"job_id" in kw.get("input", b""):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=150)
        if kw.get("input") is not None:
            return _cp(0)
        return _cp(0)

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    r = sync.run_job(TGT, sync.JobSpec(**SPEC), "~/kda-ascend", "infra/remote/runner.py",
                     task, tmp_path / "o", repo_root=repo)
    assert r["ok"] is False and r["stage"] == "job.json"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
