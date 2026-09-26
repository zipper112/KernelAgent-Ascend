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
        from harness.control.memory import IterationMemory
        self.mem = IterationMemory(self.task)   # 三源记忆层（Humanize 存续/KDA 证据/比赛经验）
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
        # 上下文预算护栏（用户上限 250k，0.8x=200k 软线）：注入量按余量自适应
        budget = int(self.models._defaults.get("context_window", 250000))
        soft = int(budget * 0.8)
        used = self._context_used_tokens()
        per_skill = max(1200, (soft - used) // 4)   # 注入侧最多吃 1/4 余量，下限 1200 字符
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
                                      "excerpt": body[:per_skill]})
        self.ev.log_audit("harness", "skill-inject", round_=round_,
                          detail={"n": len(out["skills"]),
                                  "ids": [s["id"] for s in out["skills"]],
                                  "ctx_budget": {"window": budget, "soft": soft,
                                                 "used_est": used}})
        return out

    def _context_used_tokens(self) -> int:
        """上下文占用估算：usage 账本最近 prompt_tokens（现行循环每轮独立 messages，
        不累积会话——占用=单轮 prompt 规模；此值为注入侧的自适应依据）。"""
        p = self.task / "run" / "usage.jsonl"
        if not p.exists():
            return 0
        last = ""
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = line
        return int(json.loads(last).get("prompt_tokens", 0)) if last else 0

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

    def _preflight(self, code: str, cid: str) -> str | None:
        """写后即检（零 NPU 成本）：语法编译 + CPU 桩冒烟。返回错误串或 None。
        远端 verify 一轮 ~10s 且占卡——低级错必须在本地拦截（k1-eager-norm 教训：
        6 轮里 4 轮死于 list index/接口错，全可本地拦截）。
        Triton DSL 特例：本地无 triton（只在远端容器），降级为语法检查 +
        triton 桩（kernel 体不本地执行）——执行级验证交给远端 verify。"""
        cdir = self.task / "solution" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.py").write_text(code, encoding="utf-8")
        try:
            compile(code, f"{cid}.py", "exec")
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        has_triton = re.search(r"^\s*import triton|^\s*from triton", code, re.M)
        try:
            import types
            import torch
            saved_npu = getattr(torch, "npu", None)
            saved_meth = getattr(torch.Tensor, "npu", None)
            saved_tnpu = sys.modules.get("torch_npu")
            torch.npu = types.SimpleNamespace(set_device=lambda i: None,
                                              synchronize=lambda: None,
                                              Event=type("E", (), {"__init__": lambda s, enable_timing=False: None,
                                                                   "record": lambda s: None,
                                                                   "elapsed_time": lambda s, o: 1.0}))
            torch.Tensor.npu = lambda self: self
            sys.modules["torch_npu"] = types.ModuleType("torch_npu")
            if has_triton:
                # triton 桩：jit 装饰器透传、language 属性宽容——只验 Python 层可导入性
                tri = types.ModuleType("triton")
                tri.jit = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))
                lang = types.ModuleType("triton.language")
                class _AnyConst:
                    def __getattr__(self, n):
                        return n
                lang.constexpr = "constexpr"
                lang.__getattr__ = lambda n: n        # tl.program_id/load/store 等
                tri.language = lang
                sys.modules["triton"] = tri
                sys.modules["triton.language"] = lang
            try:
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location(cid, cdir / "candidate.py")
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                spec2 = _ilu.spec_from_file_location("ref", self.task / "reference.py")
                refm = _ilu.module_from_spec(spec2)
                spec2.loader.exec_module(refm)
                import yaml as _y
                wls = _y.safe_load((self.task / "bench" / "workloads.yaml")
                                   .read_text(encoding="utf-8"))["workloads"]
                if not has_triton:      # triton 桩下 kernel() 必假失败——语义验证交远端
                    for wl in wls[:2]:
                        axes = wl["axes"]
                        x = torch.randn(axes["batch"], axes["seq"], axes["hidden"],
                                        dtype=torch.float32).to(torch.bfloat16)
                        out = mod.kernel([x])
                        if out is None:
                            return "kernel returned None"
                        ref = refm.reference([x.clone()])
                        if tuple(out.shape) != tuple(ref.shape):
                            return f"shape {tuple(out.shape)} != ref {tuple(ref.shape)}"
            finally:
                if saved_npu is not None:
                    torch.npu = saved_npu
                if saved_meth is not None:
                    torch.Tensor.npu = saved_meth
                if saved_tnpu is not None:
                    sys.modules["torch_npu"] = saved_tnpu
                else:
                    sys.modules.pop("torch_npu", None)
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:150]}"
        return None

    def write_candidate(self, round_: int, research: dict, feedback: str) -> dict:
        prev = self._candidates_summary()
        cid = f"c{len(self.ev.load_solutions()) + 1:03d}"
        # 同向 refine 复用 id（协议候选 id 规则）；**换向必须新开 id**——否则新方向
        # 覆盖旧目录，最优版本丢失（K8-Triton 实战：606us 版被后续换向轮覆盖）。
        if feedback.startswith("REVISE:"):
            sols = self.ev.load_solutions()
            last_dir = str(sols[-1].get("direction", ""))[:40] if sols else ""
            # feedback 里带方向名（writer 拿到的 FUSE/REVISE 文本含旧方向）时对比；
            # 无法判断时保守开新 id（宁可多目录不丢版本）
            if last_dir and last_dir in feedback:
                cid = sols[-1]["candidate_id"]
        skills_txt = "\n\n".join(
            f"### skill {s['id']}（{s['skill']}）\n{s['excerpt']}" for s in research["skills"]) or "（router 未命中）"
        import yaml as _y
        _dsl = _y.safe_load((self.task / "task.yaml").read_text(encoding="utf-8")) \
                .get("contract", {}).get("allowed", {}).get("dsl", ["pure-torch"])
        if "triton-ascend" in _dsl:
            iface = ("kernel(inputs: list[Tensor]) -> Tensor；inputs 与 reference.py 一致；"
                     "triton 3.5.0 + triton_ascend 3.2.2 + torch 2.10（容器内）。"
                     "Triton 源码写在 candidate.py 顶部，kernel() 内调用编译好的 triton kernel；"
                     "首次调用 JIT 编译即可（warmup 会覆盖）。")
        else:
            iface = ("kernel(inputs: list[Tensor]) -> Tensor；inputs 与 reference.py 一致；"
                     "纯 torch_npu（torch 2.7.1+torch_npu 2.7.1，无 triton）。")
        prompt = f"""你是 Ascend NPU kernel 优化 agent。任务契约见下。请基于【调研材料】写候选 kernel。

## 任务契约（摘）
{(self.task / 'task.yaml').read_text(encoding='utf-8')[:1200]}

## workload（bench/workloads.yaml）
{(self.task / 'bench' / 'workloads.yaml').read_text(encoding='utf-8')[:600]}

## 接口约定（必须遵守）
{iface}

## 调研材料（router 命中知识，必须利用；未命中则声明盲区）
### router 结果
{research['router'][:800]}
### 生产代码索引
{research['production'][:600]}
### skill 切片
{skills_txt[:4000]}

## 候选历史（learn from evidence，勿重复已否决方向）
{prev}

## 迭代记忆（上一轮完整档案——评审原文含修复线索，代码含可复用部分）
{self._memory_block()}

## 上一轮评审反馈
{feedback or '（首轮）'}

## 输出格式（分隔符协议——代码不转义，防截断浪费）
先输出三行元信息，然后代码块，总共严格控制在 250 行以内（精炼优先，注释从简）：
DIRECTION: <方向名>
HYPOTHESIS: <一句话假设>
KNOWLEDGE: <引用的 skill id 或 production 条目，逗号分隔>
===CODE===
<candidate.py 完整内容，纯文本不转义>
===END===
"""
        raw = self.models.chat("writer", [{"role": "user", "content": prompt}],
                               temperature=0.2, max_tokens=None,
                               purpose=f"round{round_}-write")
        d = self._parse_candidate(raw)
        if d is None:
            # 截断/畸形重试一次：要求只补代码（temperature 0 保一致性）
            raw2 = self.models.chat(
                "writer",
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": raw[:12000]},
                 {"role": "user", "content": "输出不完整或格式不符。重新按分隔符协议输出完整候选（DIRECTION/HYPOTHESIS/KNOWLEDGE 三行 + ===CODE=== 块），250 行内。"}],
                temperature=0.0, max_tokens=None, purpose=f"round{round_}-write-retry")
            d = self._parse_candidate(raw2)
        if d is None or "code" not in d:
            # 解析失败不再炸循环：记审计 + 返回错误占位候选（本轮 review 判 REVISE，下轮带反馈重写）
            self.ev.log_audit("harness", "write-parse-fail", target=cid, round_=round_,
                              detail={"raw_head": (raw if d is None else raw2)[:150]})
            return {"cid": cid, "direction": "parse-failed", "hypothesis":
                    "writer 输出无法解析（截断/畸形）——需要更紧凑的代码输出", "code": "", "parse_failed": True}
        # 写后即检（本地零 NPU 成本拦截低级错；失败带错误回炉重写一次）
        err = self._preflight(d["code"], cid)
        if err:
            self.ev.log_audit("harness", "preflight-reject", target=cid, round_=round_,
                              detail={"error": err[:120]})
            fix_prompt = (f"你上一版候选在本地预检就失败：{err}。"
                          "修复这个问题（大概率是接口/运行时错误，不是算法问题），"
                          "输出同格式 JSON。上版代码：\n" + d["code"][:6000])
            raw3 = self.models.chat("writer",
                                    [{"role": "user", "content": prompt},
                                     {"role": "user", "content": fix_prompt}],
                                    temperature=0.0, max_tokens=None,
                                    purpose=f"round{round_}-write-fix")
            d3 = self._parse_candidate_json(raw3)
            if d3 and "code" in d3:
                err2 = self._preflight(d3["code"], cid)
                if not err2:
                    d = d3
                else:
                    self.ev.log_audit("harness", "preflight-reject", target=cid, round_=round_,
                                      detail={"error": err2[:120], "attempt": 2})
            else:
                self.ev.log_audit("harness", "preflight-reject", target=cid, round_=round_,
                                  detail={"error": "fix-parse-failed", "attempt": 2})
        cdir = self.task / "solution" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.py").write_text(d["code"], encoding="utf-8")
        self.ev.log_audit("agent", "candidate-write", target=cid, round_=round_,
                          detail={"direction": d.get("direction"),
                                  "knowledge_used": d.get("knowledge_used", []),
                                  "hypothesis": str(d.get("hypothesis"))[:150]})
        return {"cid": cid, **d}

    @staticmethod
    def _parse_candidate(raw: str) -> dict | None:
        """分隔符协议解析：DIRECTION/HYPOTHESIS/KNOWLEDGE 三行 + ===CODE=== ... ===END=== 块。
        代码零转义（Triton 任务的 JSON 转义会浪费 ~20% 输出预算且易截断）。"""
        if not raw or not raw.strip():
            return None
        t = raw.strip()
        # 剥外层围栏（模型偶发习惯）
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        d: dict = {}
        for key, tag in (("direction", "DIRECTION"), ("hypothesis", "HYPOTHESIS"), ("knowledge_used", "KNOWLEDGE")):
            m = re.search(rf"^{tag}:\s*(.+)$", t, re.M)
            if m:
                v = m.group(1).strip()
                d[key] = [k.strip() for k in v.split(",")] if key == "knowledge_used" else v
        # code 块：===CODE=== 到 ===END===（END 缺失=截断，取到串尾）
        m = re.search(r"===CODE===\s*\n(.*?)(?:\n===END===|\Z)", t, re.S)
        if not m:
            return None
        code = m.group(0)
        # 只保留 CODE 标记之后的内容
        code = code.split("===CODE===", 1)[1]
        code = re.sub(r"^\s*\n", "", code)
        code = re.sub(r"\n?===END===\s*$", "", code)
        if not code.strip():
            return None
        d["code"] = code
        if "direction" not in d:
            d["direction"] = "unspecified"
        return d

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

    def _memory_block(self) -> str:
        """writer prompt 的记忆段：上轮档案全文 + 证据账本 + 经验库（三源内化）。"""
        lr = self.mem.last_round()
        parts = []
        if lr:
            r, rec = lr
            parts.append(f"### 上轮（round {r}）完整档案\n"
                         f"- direction: {rec.get('direction')}\n"
                         f"- verify: {json.dumps(rec.get('verify', {}), ensure_ascii=False)[:300]}\n"
                         f"- bench: {json.dumps(rec.get('bench', {}), ensure_ascii=False)[:200]}\n"
                         f"- 评审原文:\n{str(rec.get('review_text', ''))[:2000]}")
            code = str(rec.get('code', ''))
            if code:
                parts.append(f"- 上轮代码（可增量修改，勿从零重写）:\n```\n{code[:4000]}\n```")
        parts.append(f"### bench 证据账本\n{self.mem.evidence_digest(self._best_baseline_us())}")
        parts.append(f"### 沉淀经验（BitLesson——失败教训优先吸取）\n{self.mem.lessons_digest()}")
        return "\n\n".join(parts)

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

    def _best_baseline_us(self) -> float | None:
        """reviewer 参照系：历史最优 mean_us（含 c001 基线——keep 的唯一合法参照）。"""
        import csv
        p = self.task / "docs" / "benchmark.csv"
        if not p.exists():
            return None
        best = None
        for row in csv.DictReader(p.read_text(encoding="utf-8").splitlines()):
            try:
                v = float(row["mean_us"])
            except (ValueError, KeyError):
                continue
            best = v if best is None else min(best, v)
        return best

    def review(self, round_: int, cid: str, vr: dict, br: dict | None) -> str:
        template = (REPO_ROOT / "knowledge" / "prompts" / "gate-review.md").read_text(encoding="utf-8")
        best_us = self._best_baseline_us()
        cur_us = (br or {}).get("mean_us")
        beat = None
        if best_us is not None and cur_us:
            beat = round((best_us - cur_us) / best_us * 100, 1)
        prompt = f"""你是 gate 评审（只读；证据驱动）。简化轮评审（模板节选）：
{template[:1500]}

## 本轮证据
- verify：passed={vr.get('passed')} err_ratio={[w.get('err_ratio') for w in vr.get('workloads', [])]}
- bench：{json.dumps({k: br.get(k) for k in ('mean_us', 'p50_us', 'p99_us', 'speedup')}, ensure_ascii=False) if br else '未跑（verify 未过）'}
- 候选历史：{self._candidates_summary()}
- **历史最优 mean_us = {best_us}**（含基线；本轮 {round(cur_us, 1) if cur_us else 'N/A'}，
  {'快 ' + str(beat) + '%' if beat and beat > 0 else ('慢 ' + str(abs(beat)) + '%' if beat else '无可比')}）

## 裁决规则（按数据不按声明）
- verify 未过 → REVISE（附一句修复方向）或 REJECT（方向死刑）
- verify 过且 bench 出数：
  - **mean_us 优于历史最优** → keep（这是唯一刷新纪录的合法判据；speedup 字段是相对
    运行内 oracle 的比值，不可作为 keep 依据——以 mean_us 为准）
  - 未超历史最优 → REVISE（写明差多少 μs）
- 末行必须是四选一：COMPLETE / REVISE / REJECT / STOP
输出：一段简短评审 + 末行裁决（恰好一行，在最后一行）。"""
        raw = self.models.chat("reviewer", [{"role": "user", "content": prompt}],
                               temperature=0.1, purpose=f"round{round_}-review")
        self._last_review_text = raw          # 记忆层存档用（评审原文进下轮 prompt）
        self.ev.log_audit("gate", "review", target=f"round-{round_}", round_=round_,
                          detail={"tail": raw.strip().splitlines()[-1][:80] if raw.strip() else "",
                                  "best_us": best_us, "cur_us": cur_us, "beat_pct": beat})
        last = raw.strip().splitlines()[-1].strip().upper() if raw.strip() else ""
        for v in ("COMPLETE", "REVISE", "REJECT", "STOP"):
            if last == v or last.startswith(v):
                return v
        return "REVISE"   # 解析兜底：保守打回

    # ---------- 主循环 ----------

    def run(self) -> int:
        # 断点续跑：轮号接续 state.round（不重数——监督按轮分组依赖此）；反馈恢复上轮裁决
        cur = self.st.require()
        start_round = int(cur.get("round") or 0) + 1   # 续跑从下一轮起（防轮号碰撞污染监督分组）
        if cur.get("terminal"):
            self.st.update(terminal=None)     # 复活：续跑清终态
        state = self.st
        feedback = ""
        last = self.ev.load_solutions()
        if last:
            feedback = f"{cur.get('last_verdict') or 'REVISE'}: 续跑——基于最新证据链继续（上一候选 {last[-1]['candidate_id']} dir={last[-1]['direction']}）"
        for i in range(self.max_rounds):
            round_ = start_round + i
            state = self.st.update(round=round_)
            print(f"[loop] round {round_} RESEARCH...", flush=True)
            research = self.research(round_, feedback)
            print(f"[loop] round {round_} WRITE...", flush=True)
            cand = self.write_candidate(round_, research, feedback)
            print(f"[loop] round {round_} VERIFY {cand['cid']}...", flush=True)
            if cand.get("parse_failed"):
                vr = {"passed": False, "workloads": [], "error": "write-parse-failed"}
                self.ev.log_audit("harness", "verify-step", target=cand["cid"], round_=round_,
                                  detail={"rc": 2, "passed": False, "skipped": "parse-failed"})
            else:
                vr = self.verify(round_, cand["cid"])
            br = None
            if vr.get("passed"):
                print(f"[loop] round {round_} BENCH...", flush=True)
                br = self.bench(round_, cand["cid"])
            print(f"[loop] round {round_} REVIEW...", flush=True)
            verdict = self.review(round_, cand["cid"], vr, br)
            self.st.update(last_verdict=verdict)
            feedback = f"{verdict}: {cand.get('hypothesis', '')}"
            # 记忆层轮末存档（Humanize 存续：下轮注入全文而非摘要）
            self.mem.save_round(round_, {
                "direction": cand.get("direction"), "hypothesis": cand.get("hypothesis"),
                "verify": {"passed": vr.get("passed"),
                           "err_ratio": [w.get("err_ratio") for w in vr.get("workloads", [])][:3],
                           "error": str((vr.get("workloads") or [{}])[0].get("error", ""))[:150]},
                "bench": {k: (br or {}).get(k) for k in ("mean_us", "p50_us", "p99_us", "speedup")},
                "review_text": self._last_review_text,
                "code": cand.get("code", ""),
            })
            # BitLesson 沉淀（比赛经验：每轮一条，失败教训优先）
            if verdict == "keep" and br and br.get("mean_us"):
                self.mem.add_lesson(round_, "win",
                                    f"{cand.get('direction')} mean={br['mean_us']:.0f}us 刷新最优——该方向有效")
            elif not vr.get("passed"):
                err = str((vr.get("workloads") or [{}])[0].get("error", ""))[:120]
                self.mem.add_lesson(round_, "fail",
                                    f"{cand.get('direction')} verify 挂: {err or '数值超差'}——避免同类接口/边界错误")
            # 熔断②接线（v0.2 协议）：REVISE/REJECT 连续 3 次同方向 → 强制换向注入
            if verdict in ("REVISE", "REJECT"):
                n = self.st.bump_direction_fail(str(cand.get("direction", "unknown"))[:40])
                self.ev.log_audit("harness", "fuse-check", target=f"direction={cand.get('direction', '')[:40]}",
                                  round_=round_, detail={"consecutive_fails": n})
                if n >= 3:
                    banned = cand.get("direction", "")
                    feedback = (f"FUSE-DIRECTION: 方向「{banned}」已连续 {n} 次未达标，禁止再用。"
                                f"必须换一个根本不同的优化方向（读 bench 证据找新瓶颈）。")
                    self.st.bump_direction_fail(banned[:40])   # 保持计数；下一候选新方向自动另起
                    self.ev.log_audit("harness", "fuse", target=f"direction={banned[:40]}",
                                      round_=round_, detail={"reason": "3 consecutive fails", "action": "force-switch"})
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
