"""infra/remote/sync.py —— 本地↔远端文件同步与 Job 生命周期（ADR-011 §8b）。

架构原则：本地是唯一的家（证据链/git/lock），远端 ~/kda-ascend/<task>/{payload,results}
是可再生镜像。同步走 tar-over-ssh 两跳（Windows 无原生 rsync；job spec 显式文件清单=天然增量）。

Job 生命周期：组装 job.json → push payload → 远端 runner 执行 → pull result → 本地写证据链。
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .executor import RemoteTarget

JOB_KINDS = ("verify", "bench", "profile")
VERDICTS = ("keep", "revise", "reject")


@dataclass
class JobSpec:
    """§8b job.json schema v1（组装方=本地 harness；校验=两端）。"""
    job_id: str
    kind: str                       # verify|bench|profile
    candidate_id: str
    task: str
    files: list[str]                # 相对任务根的文件清单（push 范围）
    workloads: list[dict] = field(default_factory=list)   # [{id, axes{}, dtype}]
    workload_set: str = "l0"        # l0|l1|full（校验用）
    timeout_s: int = 600
    device_id: int = 0
    dsl: str = "triton-ascend"
    extra: dict = field(default_factory=dict)   # bench: warmup/samples；profile: metric_groups

    def validate(self) -> None:
        assert self.kind in JOB_KINDS, f"kind 非法: {self.kind}"
        assert self.workload_set in ("l0", "l1", "full")
        assert self.device_id >= 0
        if self.kind == "bench":
            assert self.extra.get("warmup", 3) >= 1 and self.extra.get("samples", 5) >= 1

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "JobSpec":
        d = json.loads(text)
        spec = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        spec.validate()
        return spec


def _ssh_base(target: RemoteTarget) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", target.jump]


def _q(p: str) -> str:
    """远端路径引用：~ 前缀换成 $HOME/ 再 quote 余下部分（单引号会冻结 ~ 展开）。"""
    if p == "~":
        return "$HOME"
    if p.startswith("~/"):
        return f"$HOME/{shlex.quote(p[2:])}"
    return shlex.quote(p)


def push_files(target: RemoteTarget, local_root: Path, rel_files: list[str],
               remote_dir: str, timeout_s: int = 300) -> dict:
    """tar 打包本地文件 → 两跳 scp 解包。远端自动 mkdir。失败不抛异常（返回错误 dict）。"""
    missing = [f for f in rel_files if not (local_root / f).exists()]
    if missing:
        return {"ok": False, "error": f"本地缺失: {missing[:3]}", "pushed": 0}
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
        tar_path = tf.name
    try:
        with tarfile.open(tar_path, "w") as tar:
            for f in rel_files:
                tar.add(local_root / f, arcname=f)
        remote_tar = f"/tmp/kda_push_{Path(tar_path).stem}.tar"
        # 1) 推 tar 到 jump
        r = subprocess.run([*_ssh_base(target), "cat > " + shlex.quote(remote_tar)],
                           input=Path(tar_path).read_bytes(), capture_output=True, timeout=timeout_s)
        if r.returncode != 0:
            return {"ok": False, "error": f"jump 传输失败: {r.stderr.decode()[:200]}", "pushed": 0}
        # 2) jump→host scp + 解包
        inner = (f"timeout {timeout_s} ssh -o BatchMode=yes {target.host} "
                 f"'mkdir -p {_q(remote_dir)} && cat > {_q(remote_tar)}' "
                 f"< {shlex.quote(remote_tar)} && rm -f {shlex.quote(remote_tar)}")
        r = subprocess.run([*_ssh_base(target), inner], capture_output=True, text=True, timeout=timeout_s + 30)
        if r.returncode != 0:
            return {"ok": False, "error": f"host 传输失败: {r.stderr[:200]}", "pushed": 0}
        inner2 = (f"timeout 60 ssh -o BatchMode=yes {target.host} "
                  f"'cd {_q(remote_dir)} && tar xf {_q(remote_tar)} && rm -f {_q(remote_tar)}'")
        r = subprocess.run([*_ssh_base(target), inner2], capture_output=True, text=True, timeout=120)
        return ({"ok": True, "pushed": len(rel_files)} if r.returncode == 0
                else {"ok": False, "error": f"解包失败: {r.stderr[:200]}", "pushed": 0})
    finally:
        Path(tar_path).unlink(missing_ok=True)


def pull_files(target: RemoteTarget, remote_paths: list[str], local_dir: Path,
               timeout_s: int = 300) -> dict:
    """远端文件拉回本地：远端按父目录 tar（成员名=文件名，打平）→ stdout 两跳流回 → 本地解包。

    v0.1 修复：tar 字节直接走 ssh stdout（旧版写到 jump 的 /tmp 再读本机路径，本地永远读不到）；
    成员名打平（旧版 -C / 产生 home/ubuntu/... 嵌套，本地落点错位）。
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_tar = "/tmp/kda_pull.tar"
    files_sh = " ".join(_q(p) for p in remote_paths)
    inner = (f"timeout {timeout_s} ssh -o BatchMode=yes {target.host} "
             f"'rm -f {remote_tar}; for f in {files_sh}; do "
             f"tar rf {remote_tar} -C $(dirname \"$f\") $(basename \"$f\") 2>/dev/null; done; "
             f"cat {remote_tar} 2>/dev/null; rm -f {remote_tar}'")
    r = subprocess.run([*_ssh_base(target), inner], capture_output=True, timeout=timeout_s + 60)
    if r.returncode != 0 or not r.stdout:
        return {"ok": False, "error": f"拉取失败: {r.stderr.decode(errors='replace')[:200]}", "pulled": 0}
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
        tf.write(r.stdout)
        local_tar = Path(tf.name)
    try:
        with tarfile.open(local_tar) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            tar.extractall(local_dir, filter="data")
        return {"ok": True, "pulled": len(members)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200], "pulled": 0}
    finally:
        local_tar.unlink(missing_ok=True)


def _build_exec_cmd(target: RemoteTarget, spec: JobSpec, payload: str, results: str,
                    runner_rel: str) -> str:
    """构造目标机上的执行命令（纯字符串函数，可 dry 单测）。
    docker：设备直通 + payload/results 挂载 + 宿主驱动挂载，入口 = container_entry.sh。
    native：source cann_env 后直接 python3（v0 冒烟备用路径）。
    """
    if getattr(target, "exec_mode", "native") == "docker":
        assert target.docker_image, "docker 模式必须给 docker_image"
        entry_rel = f"{PurePosixPath(runner_rel).parent}/container_entry.sh"
        dev = spec.device_id
        return (
            f"sudo docker run --rm"
            f" --device /dev/davinci{dev} --device /dev/davinci_manager"
            f" --device /dev/devmm_svm --device /dev/hisi_hdc"
            f" -v {_q(payload)}:/work/payload:ro"
            f" -v {_q(results)}:/work/results"
            f" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver_host:ro"
            f" --entrypoint bash {shlex.quote(target.docker_image)}"
            f" {shlex.quote(f'/work/payload/{entry_rel}')}"
            f" {_q(f'/work/payload/{runner_rel}')}"
            f" --job job.json --results-dir /work/results"
        )
    pre = f"source {_q(target.cann_env)} && " if getattr(target, "cann_env", "") else ""
    return (f"{pre}python3 {_q(f'{payload}/{runner_rel}')}"
            f" --job {_q(f'{payload}/job.json')} --results-dir {_q(results)}")


def _job_json_for_remote(spec: JobSpec, docker: bool) -> str:
    """job.json 落盘内容（可单测）。docker 模式：容器内只直通一张卡，CANN 逻辑编号恒为 0
    （physical=7 直通后 valid range 是 [0,1)），物理选卡由 --device 直通表达。"""
    d = json.loads(spec.to_json())
    if docker:
        d["physical_device_id"] = d["device_id"]
        d["device_id"] = 0
    return json.dumps(d, ensure_ascii=False, indent=1)


def run_job(target: RemoteTarget, spec: JobSpec, mirror_root: str, runner_rel: str,
            task_root_local: Path, local_out_dir: Path, timeout_s: int = 900,
            repo_root: Path | None = None) -> dict:
    """完整 Job 生命周期：push(job.json+files+runner) → 远端执行 → pull(result)。

    文件两源合流进临时 staging：runner_rel/container_entry.sh 相对 repo_root，
    spec.files 相对 task_root_local；远端 payload 内保持相同相对布局。
    """
    spec.validate()
    repo_root = repo_root or task_root_local.parent.parent
    remote_task = f"{mirror_root}/{spec.task}"
    payload = f"{remote_task}/payload"
    results = f"{remote_task}/results"
    docker = getattr(target, "exec_mode", "native") == "docker"
    entry_rel = f"{PurePosixPath(runner_rel).parent}/container_entry.sh"
    files = [runner_rel, entry_rel, *spec.files]
    # staging：两源文件按远端布局摆好再打包（push 的 local_root 语义单一化）
    staging = Path(tempfile.mkdtemp(prefix="kda_stage_"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        tf.write(_job_json_for_remote(spec, docker))
        job_local = Path(tf.name)
    try:
        for f, root in [(runner_rel, repo_root), (entry_rel, repo_root), *[(f, task_root_local) for f in spec.files]]:
            src = root / f
            if not src.exists():
                return {"ok": False, "stage": "stage", "error": f"本地缺失: {f}（{root}）", "pushed": 0}
            dst = staging / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        r = push_files(target, staging, files, payload, timeout_s)
        if not r["ok"]:
            return {"ok": False, "stage": "push", **r}
        # job.json 单独走 stdin 通道；顺带预建 results（docker -v 挂载源须先存在且归 ubuntu 所有）
        r2 = subprocess.run(
            [*_ssh_base(target),
             f"timeout 120 ssh -o BatchMode=yes {target.host} 'mkdir -p {_q(payload)} {_q(results)} && cat > {_q(payload + "/job.json")}'"],
            input=job_local.read_bytes(), capture_output=True, timeout=150)
        if r2.returncode != 0:
            return {"ok": False, "stage": "job.json", "error": r2.stderr.decode()[:200]}
        # 2) 远端执行（同步大超时；Phase 1 换 nohup+轮询）
        exec_cmd = _build_exec_cmd(target, spec, payload, results, runner_rel)
        inner = (f"timeout {spec.timeout_s} ssh -o BatchMode=yes {target.host} "
                 f"{shlex.quote(exec_cmd)}")
        r3 = subprocess.run([*_ssh_base(target), inner], capture_output=True, text=True,
                            timeout=spec.timeout_s + 120)
        # 3) pull result
        r4 = pull_files(target, [f"{results}/{spec.job_id}.json"], local_out_dir, timeout_s=180)
        result = {"ok": r4["ok"], "stage": "pull" if not r4["ok"] else "done",
                  "remote_rc": r3.returncode, "remote_tail": (r3.stdout or r3.stderr)[-400:]}
        if r4["ok"]:
            p = local_out_dir / f"{spec.job_id}.json"
            if p.exists():
                result["result"] = json.loads(p.read_text(encoding="utf-8"))
        return result
    finally:
        job_local.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
