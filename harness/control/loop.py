"""harness/control/loop.py —— 自主迭代主循环（v0.1：K1 考题首发）。

循环（协议 §1 任务图 C 批的最小实装）：
  RESEARCH（router 检索 + production 代码索引 + skill 切片注入；audit 记事件）
  → WRITE（writer LLM 产候选代码，落盘 solution/<cid>/）
  → VERIFY（kda verify：远端四步协议；失败也入链计 direction_fails）
  → BENCH（verify 过才 bench；evidence 即时落盘）
  → REVIEW（reviewer LLM 证据驱动裁决 keep/revise/reject；keep 即 git commit）
  → 熔断（round 上限 / direction_fails / QuotaError→checkpoint）

监督契约（tools/supervise.py 消费）：每轮 audit 必须出现事件序列
  router-query → production-ref|blindspot → skill-inject → candidate-write → verify → [bench] → review
缺任一 = 违规，人工干预。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.core.evidence import Evidence  # noqa: E402
from harness.core.state import TaskState  # noqa: E402
from harness.models import ModelsClient, QuotaError  # noqa: E402

QUERY = REPO_ROOT / "knowledge" / "router" / "query.py"


def _sh(cmd: list[str], timeout: int = 30) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout or r.stderr


class AutonomousLoop:
    def __init__(self, task_root: Path, max_rounds: int = 3):
        self.task = Path(task_root)
        self.name = self.task.name
        self.ev = Evidence(self.task)
        self.st = TaskState(self.task, self.ev)
        self.models = ModelsClient(self.task)
        self.max_rounds = max_rounds
        self.budget = self._load_budget()

    def _load_budget(self) -> dict:
        import yaml
        cfg = yaml.safe_load((self.task / "config.yaml").read_text(encoding="utf-8"))
        return cfg.get("budget", {})

    # ---------- 阶段 1：RESEARCH（监督点：不调研不许写） ----------

    def research(self, round_: int, direction_hint: str) -> dict:
        out = {"router": "", "production": "", "skills": [], "blindspot": False}
        # 三轴检索（symptom 从 round-contract 的 direction/hypothesis 提取或任务预设）
        symptoms = self._task_symptoms()
        fam = self._task_family()
        arch = self._task_arch()
        args = [sys.executable, str(QUERY), "--compact"]
        for s in symptoms:
            args += ["--symptom", s]
        args += ["--op-family", fam, "--arch", arch]
        out["router"] = _sh(args)
        self.ev.log_audit("harness", "router-query", round_=round_,
                          detail={"symptoms": symptoms, "family": fam, "arch": arch,
                                  "hit": "NO MATCH" not in out["router"]})
        if "NO MATCH" in out["router"]:
            out["blindspot"] = True
            self.ev.log_audit("harness", "blindspot", round_=round_, detail={"query": symptoms})
        # 生产代码层
        prod = _sh([sys.executable, str(QUERY), "--production", "--compact",
                    "--op-family", fam, "--arch", arch])
        out["production"] = prod
        self.ev.log_audit("harness", "production-ref", round_=round_,
                          detail={"hit": "NO MATCH" not in prod})
        # skill 切片注入（router 命中的 ref：文件直接读；目录则找其下 SKILL.md 或首个 .md）
        for m in re.finditer(r"\[[^\]]+\] ([\w-]+): ([\w-]+) -> ([^\s]+)", out["router"]):
            skill_id, skill, ref = m.group(1), m.group(2), m.group(3)
            p = REPO_ROOT / ref
            if p.is_dir():
                cand = p / "SKILL.md"
                if not cand.exists():
                    mds = sorted(p.rglob("*.md"))
                    cand = mds[0] if mds else None
                p = cand
            if p and p.exists() and p.is_file():
                body = p.read_text(encoding="utf-8")
                out["skills"].append({"id": skill_id, "skill": skill, "ref": ref,
                                      "excerpt": body[:1500]})
        self.ev.log_audit("harness", "skill-inject", round_=round_,
                          detail={"n": len(out["skills"]),
                                  "ids": [s["id"] for s in out["skills"]]})
        return out

    def _task_symptoms(self) -> list[str]:
        import yaml
        c = yaml.safe_load((self.task / "task.yaml").read_text(encoding="utf-8"))
        return c.get("contract", {}).get("research", {}).get("symptoms",
               c.get("contract", {}).get("_symptoms", ["pipeline-bubble"]))

    def _task_family(self) -> str:
        import yaml
        c = yaml.safe_load((self.task / "task.yaml").read_text(encoding="utf-8"))
        return c.get("contract", {}).get("op_family", "elementwise")

    def _task_arch(self) -> str:
        import yaml
        cfg = yaml.safe_load((self.task / "config.yaml").read_text(encoding="utf-8"))
        return cfg.get("arch", "dav_2201")

    # ---------- 阶段 2：WRITE ----------

    def write_candidate(self, round_: int, research: dict, feedback: str) -> dict:
        prev = self._candidates_summary()
        cid = f"c{len(self.ev.load_solutions()) + 1:03d}"
        # 同一方向的 revise 复用 id（协议候选 id 规则）：feedback 是 revise 时取当前 id
        if feedback.startswith("REVISE:"):
            sols = self.ev.load_solutions()
            cid = sols[-1]["candidate_id"] if sols else cid
        skills_txt = "\n\n".join(
            f"### skill {s['id']}（{s['skill']}）\n{s['excerpt']}" for s in research["skills"]) or "（router 未命中）"
        prompt = f"""你是 Ascend NPU kernel 优化 agent。任务契约见下。请基于【调研材料】写候选 kernel。

## 任务契约（摘）
{(self.task / 'task.yaml').read_text(encoding='utf-8')[:1200]}

## workload（bench/workloads.yaml）
{(self.task / 'bench' / 'workloads.yaml').read_text(encoding='utf-8')[:600]}

## 接口约定（必须遵守）
kernel(inputs: list[Tensor]) -> Tensor；inputs 与 reference.py 一致；纯 torch_npu（torch 2.7.1+torch_npu 2.7.1，无 triton）。

## 调研材料（router 命中知识，必须利用；未命中则声明盲区）
### router 结果
{research['router'][:800]}
### 生产代码索引
{research['production'][:600]}
### skill 切片
{skills_txt[:4000]}

## 候选历史（learn from evidence，勿重复已否决方向）
{prev}

## 上一轮评审反馈
{feedback or '（首轮）'}

## 输出格式（严格 JSON，无 markdown 代码围栏）
{{"direction": "<方向名>", "hypothesis": "<一句话假设>",
  "knowledge_used": ["<引用的 skill id 或 production 条目>"],
  "code": "<candidate.py 完整内容>"}}
"""
        raw = self.models.chat("writer", [{"role": "user", "content": prompt}],
                               temperature=0.2, max_tokens=16384,
                               purpose=f"round{round_}-write")
        d = self._parse_candidate_json(raw)
        if d is None:
            # 截断/畸形重试一次：要求只补 JSON（temperature 0 保一致性）
            raw2 = self.models.chat(
                "writer",
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": raw[:12000]},
                 {"role": "user", "content": "上面的 JSON 不完整或非法。重新输出【完整】的单一 JSON 对象（code 字段内的换行用 \\n 转义），不要任何解释或 markdown 围栏。"}],
                temperature=0.0, max_tokens=16384, purpose=f"round{round_}-write-retry")
            d = self._parse_candidate_json(raw2)
        if d is None or "code" not in d:
            raise RuntimeError(f"writer 输出解析失败（两轮）：{(raw if d is None else raw2)[:200]}")
        cdir = self.task / "solution" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.py").write_text(d["code"], encoding="utf-8")
        self.ev.log_audit("agent", "candidate-write", target=cid, round_=round_,
                          detail={"direction": d.get("direction"),
                                  "knowledge_used": d.get("knowledge_used", []),
                                  "hypothesis": str(d.get("hypothesis"))[:150]})
        return {"cid": cid, **d}

    @staticmethod
    def _parse_candidate_json(raw: str) -> dict | None:
        """宽容解析：剥 markdown 围栏 → 试严格 JSON → 失败则字段级正则抽取。"""
        if not raw or not raw.strip():
            return None
        t = raw.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
            t = re.sub(r"\n?```\s*$", "", t)
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        s = m.group(0)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 字段级抽取（截断容错）：code 常在最后被截——取到最后一个完整字段
        fields = {}
        for key in ("direction", "hypothesis"):
            mm = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
            if mm:
                fields[key] = mm.group(1)
        cm = re.search(r'"code"\s*:\s*"(.*)', s, re.S)   # 截断容错：吃到串尾
        if cm and fields:
            code = cm.group(1)
            # 去掉尾部可能的未完成转义与残破碎片
            if code.endswith("\\"):
                code = code[:-1]
            try:
                fields["code"] = json.loads(f'"{code}"')
            except json.JSONDecodeError:
                fields["code"] = code.encode().decode("unicode_escape", errors="replace")
            return fields
        return None
        cdir = self.task / "solution" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.py").write_text(d["code"], encoding="utf-8")
        self.ev.log_audit("agent", "candidate-write", target=cid, round_=round_,
                          detail={"direction": d.get("direction"),
                                  "knowledge_used": d.get("knowledge_used", []),
                                  "hypothesis": str(d.get("hypothesis"))[:150]})
        return {"cid": cid, **d}

    def _candidates_summary(self) -> str:
        sols = self.ev.load_solutions()
        if not sols:
            return "（无）"
        return "\n".join(f"- {s['candidate_id']} [{s['status']}/{s.get('stage')}] dir={s['direction']}" for s in sols[-8:])

    # ---------- 阶段 3/4：VERIFY / BENCH（复用 CLI 内部逻辑） ----------

    def _run_cli(self, fn_name: str, cid: str) -> tuple[int, dict]:
        import harness.cli as cli
        args = argparse.Namespace(task=str(self.task), candidate=cid, workload_set="l0")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = getattr(cli, f"cmd_{fn_name}")(args)
        try:
            return rc, json.loads(buf.getvalue())
        except Exception:
            return rc, {"raw": buf.getvalue()[:300]}

    def verify(self, round_: int, cid: str) -> dict:
        rc, out = self._run_cli("verify", cid)
        self.ev.log_audit("harness", "verify-step", target=cid, round_=round_,
                          detail={"rc": rc, "passed": out.get("passed")})
        return out

    def bench(self, round_: int, cid: str) -> dict:
        rc, out = self._run_cli("bench", cid)
        self.ev.log_audit("harness", "bench-step", target=cid, round_=round_,
                          detail={"rc": rc, "mean_us": out.get("mean_us"),
                                  "speedup": out.get("speedup")})
        return out

    # ---------- 阶段 5：REVIEW ----------

    def review(self, round_: int, cid: str, vr: dict, br: dict | None) -> str:
        template = (REPO_ROOT / "knowledge" / "prompts" / "gate-review.md").read_text(encoding="utf-8")
        prompt = f"""你是 gate 评审（只读；证据驱动）。简化轮评审（模板节选）：
{template[:1500]}

## 本轮证据
- verify：passed={vr.get('passed')} err_ratio={[w.get('err_ratio') for w in vr.get('workloads', [])]}
- bench：{json.dumps({k: br.get(k) for k in ('mean_us', 'p50_us', 'p99_us', 'speedup')}, ensure_ascii=False) if br else '未跑（verify 未过）'}
- 候选历史：{self._candidates_summary()}

## 裁决规则
- verify 未过 → REVISE（附一句修复方向）或 REJECT（方向死刑）
- verify 过且 bench 出数 → 按证据判 keep（继续此方向）/ REVISE（可改进）
- 末行必须是四选一：COMPLETE / REVISE / REJECT / STOP
输出：一段简短评审 + 末行裁决（恰好一行，在最后一行）。"""
        raw = self.models.chat("reviewer", [{"role": "user", "content": prompt}],
                               temperature=0.1, purpose=f"round{round_}-review")
        self.ev.log_audit("gate", "review", target=f"round-{round_}", round_=round_,
                          detail={"tail": raw.strip().splitlines()[-1][:80] if raw.strip() else ""})
        last = raw.strip().splitlines()[-1].strip().upper() if raw.strip() else ""
        for v in ("COMPLETE", "REVISE", "REJECT", "STOP"):
            if last == v or last.startswith(v):
                return v
        return "REVISE"   # 解析兜底：保守打回

    # ---------- 主循环 ----------

    def run(self) -> int:
        feedback = ""
        for round_ in range(1, self.max_rounds + 1):
            state = self.st.update(round=round_)
            print(f"[loop] round {round_} RESEARCH...", flush=True)
            research = self.research(round_, feedback)
            print(f"[loop] round {round_} WRITE...", flush=True)
            cand = self.write_candidate(round_, research, feedback)
            print(f"[loop] round {round_} VERIFY {cand['cid']}...", flush=True)
            vr = self.verify(round_, cand["cid"])
            br = None
            if vr.get("passed"):
                print(f"[loop] round {round_} BENCH...", flush=True)
                br = self.bench(round_, cand["cid"])
            print(f"[loop] round {round_} REVIEW...", flush=True)
            verdict = self.review(round_, cand["cid"], vr, br)
            feedback = f"{verdict}: {cand.get('hypothesis', '')}"
            if verdict in ("COMPLETE", "STOP"):
                self.st.update(terminal=verdict)
                print(f"[loop] 终局：{verdict}", flush=True)
                return 0
            if verdict == "keep":
                subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, capture_output=True)
                subprocess.run(["git", "commit", "-m",
                                f"keep({cand['cid']}): auto {cand.get('direction')}"],
                               cwd=REPO_ROOT, capture_output=True)
        self.st.update(terminal="MAXITER")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()
    try:
        return AutonomousLoop(REPO_ROOT / "tasks" / args.task, args.max_rounds).run()
    except QuotaError as e:
        print(f"[loop] 配额暂停（状态已保存）：{e}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
