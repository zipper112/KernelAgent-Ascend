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
from harness.core.state import TaskState, is_pseudo_direction  # noqa: E402
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
        self._last_review_text = ""             # P0-7：评审原文属性预置（异常路径不 AttributeError）
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
        # 本地无 torch（jump ka 环境纯 harness）→ 降级纯语法检查：
        # torch 语义验证交给远端 chained verify（e15 容器内 torch 完整）。
        # 曾因强跑 CPU 桩（ModuleNotFoundError: torch）浪费 writer 12k token 修不存在的错。
        try:
            import torch  # noqa: F401
        except ImportError:
            return None
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
        # P0-2：parent 链计算。同向 refine（REVISE 且方向承接上轮）→ 上轮 cid 继承
        # 上下文但**不**复用 id（id 复用会覆盖最优版本——K8 实战教训）；换向 → 全局最优 keep；
        # 首个候选 → None（根）。
        sols = self.ev.load_solutions()
        latest = self.ev.latest_status_map()
        parent_cid = self._parent_for(sols, latest, feedback)
        cid = f"c{len(latest) + 1:03d}"
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

        ## NPU 访问上下文（===EXEC=== 块的写作依据；canonical runner 是唯一出数口径）
        {self._access_context()}

        ## 上一轮评审反馈
        {feedback or '（首轮）'}

        ## 输出格式（分隔符协议——代码/命令不转义，防截断浪费）
        先输出三行元信息，然后代码块与 EXEC 块，总共严格控制在 260 行以内（精炼优先，注释从简）：
DIRECTION: <方向名>
HYPOTHESIS: <一句话假设>
KNOWLEDGE: <引用的 skill id 或 production 条目，逗号分隔>
===CODE===
<candidate.py 完整内容，纯文本不转义>
===END===
===EXEC===
<远端验证/测量的 shell 命令序列，按 NPU 访问上下文的 4 步义务写；
 可加前置探针，但出数必须走 canonical runner；留空则 harness 用 canonical 模板兜底>
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
                    "writer 输出无法解析（截断/畸形）——需要更紧凑的代码输出", "code": "",
                    "parse_failed": True, "parent": parent_cid}
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
                                  "hypothesis": str(d.get("hypothesis"))[:150],
                                  "parent": parent_cid})
        return {"cid": cid, "parent": parent_cid, **d}

    @staticmethod
    def _parse_candidate(raw: str) -> dict | None:
        """分隔符协议解析：DIRECTION/HYPOTHESIS/KNOWLEDGE 三行 + ===CODE=== 块 +
        可选 ===EXEC=== 块（ADR-013：LLM 自主远端访问命令序列）。
        代码/EXEC 零转义（Triton 任务的 JSON 转义会浪费 ~20% 输出预算且易截断）。"""
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
        # EXEC 块（可选）：与 CODE 同族解析；缺省=canonical 兜底
        me = re.search(r"===EXEC===\s*\n(.*?)(?:\n===END===|\Z)", t, re.S)
        if me:
            d["exec"] = me.group(1).strip()
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

    @staticmethod
    def _parent_for(sols: list[dict], latest: dict[str, dict], feedback: str) -> str | None:
        """P0-2 DAG parent：REVISE 反馈含上轮方向名（同向 refine）→ 上轮 cid；
        其余（换向/首轮/FUSE）→ 全局最优 keep cid；无 keep → None。"""
        if not sols:
            return None
        last = sols[-1]
        last_dir = str(last.get("direction", ""))[:40]
        if (feedback.startswith("REVISE:") or feedback.startswith("REJECT:")) \
                and last_dir and last_dir in feedback:
            return last["candidate_id"]
        keeps = [r for r in latest.values() if r.get("status") == "keep"]
        if keeps:
            return max(keeps, key=lambda r: int(r.get("round") or 0))["candidate_id"]
        return None

    def _access_context(self) -> str:
        """ADR-013：NPU 访问手册（context.py 生成；remote 未启用时给本地占位）。"""
        import yaml
        cfg = yaml.safe_load((self.task / "config.yaml").read_text(encoding="utf-8"))
        rem = cfg.get("execution", {}).get("remote", {})
        if not rem.get("enabled", False):
            return "（remote 未启用——本地执行模式）"
        from harness.context import access_manual
        manual = access_manual(self.task, rem)
        # 手册同步落盘（人可查 + 溯源）
        (self.task / "docs").mkdir(exist_ok=True)
        (self.task / "docs" / "npu-access.md").write_text(manual, encoding="utf-8")
        return manual

    def _memory_block(self) -> str:
        """writer prompt 的记忆段：上轮档案全文 + 证据账本 + 经验库（三源内化）。
        P0-6：评审原文截断自适应——min(6000, 上下文余量/6)，至少 2500 字
        （完整保留裁决+逐条 issue；此前硬 2000 截断会切掉修复线索）。"""
        lr = self.mem.last_round()
        parts = []
        budget = int(self.models._defaults.get("context_window", 250000))
        soft = int(budget * 0.8)
        used = self._context_used_tokens()
        cap = max(2500, min(6000, (soft - used) // 6))
        if lr:
            r, rec = lr
            parts.append(f"### 上轮（round {r}）完整档案\n"
                         f"- direction: {rec.get('direction')}\n"
                         f"- verify: {json.dumps(rec.get('verify', {}), ensure_ascii=False)[:300]}\n"
                         f"- bench: {json.dumps(rec.get('bench', {}), ensure_ascii=False)[:200]}\n"
                         f"- 评审原文:\n{str(rec.get('review_text', ''))[:cap]}")
            code = str(rec.get('code', ''))
            if code:
                parts.append(f"- 上轮代码（可增量修改，勿从零重写）:\n```\n{code[:4000]}\n```")
        parts.append(f"### 近 3 轮趋势（方向×耗时×结果）\n{self.mem.trend_digest(3)}")
        parts.append(f"### bench 证据账本\n{self.mem.evidence_digest(self._best_baseline_us())}")
        parts.append(f"### 沉淀经验（BitLesson——win 优先，最多 5 条）\n{self.mem.lessons_digest()}")
        return "\n\n".join(parts)

    def _candidates_summary(self) -> str:
        sols = self.ev.load_solutions()
        if not sols:
            return "（无）"
        return "\n".join(f"- {s['candidate_id']} [{s['status']}/{s.get('stage')}] dir={s['direction']}" for s in sols[-8:])

    # ---------- 阶段 3/4：EXEC（ADR-013：LLM 自主远端访问，canonical 兜底） ----------

    def _remote_cfg(self) -> dict:
        import yaml
        return yaml.safe_load((self.task / "config.yaml")
                              .read_text(encoding="utf-8")).get("execution", {}).get("remote", {})

    def _gen_jobs(self, candidate_id: str) -> dict:
        """生成 verify+bench 两个 job.json（workload 双档/chained/sha 全在 harness 侧——
        测量纪律不交给 LLM）。落盘 run/job-verify.json / run/job-bench.json 并返回
        {verify: job, bench: job, files: 同步清单, shas}。"""
        import yaml
        from harness.context import build_job_payload, payload_files, _sha8
        wls_data = yaml.safe_load((self.task / "bench" / "workloads.yaml")
                                  .read_text(encoding="utf-8"))["workloads"]
        import harness.cli as cli
        dev_wls = []
        for w in cli._filter_workloads(self.task, wls_data, "l0"):
            item = {"id": w["id"], "axes": w["axes"], "dtype": w.get("dtype", "fp16")}
            if "inputs" in w:
                item["inputs"] = w["inputs"]
            dev_wls.append(item)
        full_wls = []
        for w in cli._filter_workloads(self.task, wls_data, "full"):
            item = {"id": w["id"], "axes": w["axes"], "dtype": w.get("dtype", "fp16")}
            if "inputs" in w:
                item["inputs"] = w["inputs"]
            full_wls.append(item)
        tcfg = yaml.safe_load((self.task / "task.yaml").read_text(encoding="utf-8"))
        vextra = {}
        if (tcfg.get("contract", {}) or {}).get("verify_mode") == "chained":
            vextra = {"verify_mode": "chained",
                      "chain_steps": int(tcfg["contract"].get("chain_steps", 3))}
        meas = yaml.safe_load((self.task / "config.yaml")
                              .read_text(encoding="utf-8")).get("measurement", {})
        bextra = {"warmup": meas.get("warmup", 3), "samples": meas.get("samples", 5)}
        rem = self._remote_cfg()
        dev = int(rem.get("device_id", 0))
        vjob = build_job_payload(self.task, candidate_id, "verify", dev_wls, vextra, dev)
        bjob = build_job_payload(self.task, candidate_id, "bench", dev_wls, bextra, dev)
        fjob = dict(bjob)     # full 档复核用（COMPLETE 前）
        fjob["workloads"] = full_wls
        fjob["workload_set"] = "full"
        (self.task / "run").mkdir(exist_ok=True)
        for name, job in (("job-verify.json", vjob), ("job-bench.json", bjob), ("job-bench-full.json", fjob)):
            (self.task / "run" / name).write_text(
                json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        shas = {}
        for f in ("reference.py", "bench/workloads.yaml"):
            p = self.task / f
            if p.exists():
                shas[f] = _sha8(p)
        return {"verify": vjob, "bench": bjob, "bench_full": fjob,
                "files": payload_files(self.task, candidate_id), "shas": shas}

    def _canonical_exec(self, kind: str, jobs: dict, candidate_id: str = "c001") -> str:
        """canonical 兜底 EXEC 脚本（LLM 未给 ===EXEC=== 块时用）。
        tar 多次 -C 拍平到 payload 期望布局（与 cli._run_remote_job 同构）。"""
        from harness.context import canonical_cmd, remote_workspace
        rem = self._remote_cfg()
        host = rem.get("host", "yq-e15")
        ws = remote_workspace(self.task.name, rem)
        job = jobs[kind]
        job_id = job["job_id"]
        frozen = [f for f in ("reference.py", "bench/workloads.yaml")
                  if (self.task / f).exists()]
        tar_args = ("-C . infra/remote/runner.py infra/remote/container_entry.sh "
                    f"-C tasks/{self.task.name} solution/{candidate_id}/candidate.py "
                    + " ".join(frozen))
        jname = "job-bench-full.json" if kind == "bench_full" else f"job-{kind}.json"
        return (
            f"tar cf - {tar_args} | ssh {host} 'mkdir -p {ws}/payload {ws}/results && tar xf - -C {ws}/payload'\n"
            f"cat tasks/{self.task.name}/run/{jname} | ssh {host} 'cat > {ws}/payload/job.json'\n"
            f"ssh {host} \"{canonical_cmd(job, rem)}\"\n"
            f"ssh {host} 'cat {ws}/results/{job_id}.json'\n"
        )

    def _run_exec(self, round_: int, script: str, jobs: dict, kind: str,
                  candidate_id: str) -> dict:
        """执行 EXEC 脚本（policy 已过）→ 解析 results json。失败返回 {ok:False,...}。"""
        host = self._remote_cfg().get("host", "yq-e15")
        # 记录脚本供审计/复盘
        (self.task / "run").mkdir(exist_ok=True)
        (self.task / f"run/round-{round_}-exec.sh").write_text(script, encoding="utf-8")
        try:
            # 脚本在仓根 cwd 下执行（canonical 模板里的 tar -C ./cat tasks/... 均为仓内相对路径）
            proc = subprocess.run(["bash", "-s"], input=script, capture_output=True,
                                  text=True, timeout=700, cwd=str(REPO_ROOT))
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "exec", "error": "exec-timeout>700s"}
        job = jobs[kind]
        # 从 stdout 提取 results json（最后一行合法 JSON 且含 job_id）
        result = None
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and '"job_id"' in line:
                try:
                    d = json.loads(line)
                    if d.get("job_id") == job["job_id"]:
                        result = d
                        break
                except json.JSONDecodeError:
                    continue
        if result is None:
            # 单行提取失败→尝试整体最大 JSON 块
            try:
                i = (proc.stdout or "").rindex('{"job_id"')
                result = json.loads(proc.stdout[i:proc.stdout.find("}", i) + 1]) \
                    if False else None
            except ValueError:
                result = None
        self.ev.log_audit("harness", "exec-run", target=candidate_id, round_=round_,
                          detail={"kind": kind, "rc": proc.returncode,
                                  "stdout_tail": (proc.stdout or "")[-300:],
                                  "stderr_tail": (proc.stderr or "")[-200:]})
        if result is None:
            return {"ok": False, "stage": "parse", "rc": proc.returncode,
                    "error": f"results json 未在 stdout 中（rc={proc.returncode}）",
                    "stdout_tail": (proc.stdout or "")[-300:],
                    "stderr_tail": (proc.stderr or "")[-200:]}
        return {"ok": True, "result": result}

    def exec_stage(self, round_: int, cand: dict, kind: str = "verify") -> dict:
        """EXEC 阶段入口：job 生成 → policy 检查 → 执行（LLM 脚本或 canonical 兜底）。
        返回 canonical results dict（verify: {passed, workloads}；bench: {workloads...}）。"""
        from harness.control.exec_policy import check_exec_block
        jobs = self._gen_jobs(cand["cid"])
        self.ev.log_audit("harness", f"job-spec-{kind.split('_')[0]}", target=cand["cid"],
                          round_=round_,
                          detail={"n_workloads": len(jobs[kind]["workloads"]),
                                  "shas": jobs["shas"], "job_id": jobs[kind]["job_id"],
                                  **({"verify_mode": "chained"}
                                     if jobs["verify"].get("extra", {}).get("verify_mode")
                                     else {})})
        script = cand.get("exec") or self._canonical_exec(kind, jobs, cand["cid"])
        if cand.get("exec"):
            ok, reason = check_exec_block(cand["exec"])
            if not ok:
                self.ev.log_audit("harness", "exec-policy-reject", target=cand["cid"],
                                  round_=round_, detail={"reason": reason})
                return {"ok": False, "stage": "policy", "error": reason}
        r = self._run_exec(round_, script, jobs, kind, cand["cid"])
        if not r.get("ok"):
            self.ev.log_audit("harness", "exec-fail", target=cand["cid"], round_=round_,
                              detail={"stage": r.get("stage"), "error": str(r.get("error"))[:150]})
            return r
        return r["result"]

    # ---------- 阶段 5：REVIEW ----------

    def _best_baseline_us(self) -> float | None:
        """reviewer 参照系：历史最优 mean_us（含 c001 基线——keep 的唯一合法参照）。
        P0-5：只认 phase=bench、verdict∈{benched,keep}、mean_us>0 且有限的行——
        invalid 行（mean=0/NaN）与 verify 阶段行不入参照系。"""
        import csv
        import math
        p = self.task / "docs" / "benchmark.csv"
        if not p.exists():
            return None
        best = None
        for row in csv.DictReader(p.read_text(encoding="utf-8").splitlines()):
            if row.get("phase") != "bench":
                continue
            if row.get("verdict") not in ("benched", "keep"):
                continue
            try:
                v = float(row["mean_us"])
            except (ValueError, KeyError):
                continue
            if not math.isfinite(v) or v <= 0:
                continue
            best = v if best is None else min(best, v)
        return best

    def review(self, round_: int, cid: str, vr: dict, br: dict | None) -> str:
        """P0-8：注入完整 gate 评审契约（弃 1500 字符截断——简化契约=评审质量塌陷源）。
        P1-5：每 5 轮对齐轮（round%5==0）——评审权扩至停滞检测（可判 STOP）+
        历史全轮趋势注入（Humanize full-alignment-review 机制）。"""
        template = (REPO_ROOT / "knowledge" / "prompts" / "gate-review.md").read_text(encoding="utf-8")
        alignment = (round_ % 5 == 0)
        align_block = ""
        if alignment:
            align_block = f"""
## 全量对齐审计（本轮 round {round_} = 5 的倍数）
除常规评审外你还必须：
A. 对比近几轮趋势（下表）——同一方向是否多轮无实质进展？同一问题是否反复出现？
B. 若判定停滞：末行输出 STOP（附 bench 表/具名瓶颈/已试方向清单三要素）。
### 近轮趋势
{self.mem.trend_digest(5)}
"""
        best_us = self._best_baseline_us()
        cur_us = (br or {}).get("mean_us") if (br or {}).get("valid", True) else None
        beat = None
        if best_us and cur_us:
            beat = round((best_us - cur_us) / best_us * 100, 1)
        prompt = f"""你是 gate 评审（只读；证据驱动）。以下是完整评审契约与模板，逐条遵守：

{template}
{align_block}

## 本轮证据（你的裁决只认这些，不认 agent 声明）
- 候选：{cid}（direction 见 solutions.jsonl 本轮行）
- verify：passed={vr.get('passed')} err_ratio={[w.get('err_ratio') for w in vr.get('workloads', [])][:5]}
- bench：{json.dumps({k: br.get(k) for k in ('mean_us', 'p50_us', 'p99_us', 'speedup', 'valid')}, ensure_ascii=False) if br else '未跑（verify 未过）'}
- 候选历史：{self._candidates_summary()}
- **历史最优 mean_us = {best_us}**（含基线；本轮 {round(cur_us, 1) if cur_us else 'N/A'}，
  {'快 ' + str(beat) + '%' if beat and beat > 0 else ('慢 ' + str(abs(beat)) + '%' if beat else '无可比')}）

## 裁决规则（按数据不按声明；机器漂移检测会复核你的裁决）
- verify 未过 → REVISE（附一句修复方向）或 REJECT（方向死刑）
- verify 过且 bench 出数（valid）：
  - **mean_us 优于历史最优** → COMPLETE（本轮即新纪录；这是唯一刷新纪录的合法判据；
    speedup 字段是相对运行内 oracle 的比值，不可作为判据——以 mean_us 为准）
  - 未超历史最优 → REVISE（写明差多少 μs）
  - bench 无效（valid=false）→ REVISE（指出测量无效原因）
- STOP 为终局建议：必须同时给出 ①最近 bench 证据表 ②具名瓶颈 ③已试方向清单及各自失败
  证据，三要素缺一不可——否则降级为 REVISE。
- 末行必须是四选一：COMPLETE / REVISE / REJECT / STOP（恰好一行，在最后一行）。
输出结构：简短逐项评审（含 MAINLINE_GAPS/BLOCKING/QUEUED 三车道）+ 末行裁决。"""
        raw = self.models.chat("reviewer", [{"role": "user", "content": prompt}],
                               temperature=0.1, purpose=f"round{round_}-review")
        self._last_review_text = raw          # 记忆层存档用（评审原文进下轮 prompt）
        self.ev.log_audit("gate", "review", target=f"round-{round_}", round_=round_,
                          detail={"tail": raw.strip().splitlines()[-1][:80] if raw.strip() else "",
                                  "best_us": best_us, "cur_us": cur_us, "beat_pct": beat,
                                  **({"alignment": True} if alignment else {})})
        last = raw.strip().splitlines()[-1].strip().upper() if raw.strip() else ""
        for v in ("COMPLETE", "REVISE", "REJECT", "STOP"):
            if last == v or last.startswith(v):
                return v
        return "REVISE"   # 解析兜底：保守打回

    # ---------- 主循环 ----------

    def _write_round_contract(self, round_: int, cand: dict, br: dict | None) -> None:
        """P0-6：WRITE 成功后由 harness 生成本轮契约（Humanize round-contract 机制）。
        direction/hypothesis 取自 writer 自述；success_criteria 锚定当前历史最优——
        research 方向读取与 cli._current_direction 的第一源。"""
        best = self._best_baseline_us()
        target = f"mean_us < {best:.1f}（当前历史最优，dev 集）" if best else "首个可测基线（verify 过 + bench 出数）"
        # direction 行必须裸写（cli._current_direction 按 "direction:" 前缀解析）
        text = (f"# round-{round_} 契约（harness 自动生成）\n\n"
                f"direction: {cand.get('direction')}\n"
                f"hypothesis: {cand.get('hypothesis')}\n"
                f"candidate: {cand.get('cid')}\n"
                f"parent: {cand.get('parent') or '（根）'}\n"
                f"success_criteria: {target}\n"
                f"blocking: 上轮评审要点见 run/memory/round-{max(round_ - 1, 0)}.json\n")
        (self.task / "run" / f"round-{round_}-contract.md").write_text(text, encoding="utf-8")

    def _drift_check(self, round_: int, verdict: str, br: dict | None) -> str:
        """P0-3：机器漂移判定（Humanize ADVANCED/STALLED/REGRESSED 的数据版）。
        判据全部来自 benchmark.csv——不采信 LLM 自评，防 reward hacking：
        - ADVANCED：本轮 mean_us 创新最优（含首轮基线）
        - REGRESSED：有 bench 且 mean 劣于当前历史最优（本轮行计入前的最优）>5%
        - STALLED：其余（无 bench / 未超最优但差距 ≤5% / verify 挂）
        返回 progress 并写 stall_count；≥2 REPLAN 注入、≥3 stop-drift 终态由调用方处理。"""
        import csv
        p = self.task / "docs" / "benchmark.csv"
        cur = (br or {}).get("mean_us") if (br or {}).get("valid", (br is not None)) else None
        hist_best = None
        if p.exists():
            for row in csv.DictReader(p.read_text(encoding="utf-8").splitlines()):
                if row.get("phase") != "bench" or row.get("verdict") not in ("benched", "keep"):
                    continue
                try:
                    v = float(row["mean_us"])
                except (ValueError, KeyError):
                    continue
                if v > 0 and (hist_best is None or v < hist_best):
                    hist_best = v
        if cur and cur > 0 and (hist_best is None or cur < hist_best):
            progress = "ADVANCED"
        elif cur and hist_best and cur > hist_best * 1.05:
            progress = "REGRESSED"
        else:
            progress = "STALLED"
        stall = self.st.bump_stall(progress)
        self.ev.log_audit("harness", "drift-check", target=f"round-{round_}", round_=round_,
                          detail={"progress": progress, "stall_count": stall,
                                  "cur_us": cur, "hist_best_us": hist_best,
                                  "review_verdict": verdict})
        return progress

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
        try:
            for i in range(self.max_rounds):
                round_ = start_round + i
                state = self.st.update(round=round_)
                print(f"[loop] round {round_} RESEARCH...", flush=True)
                research = self.research(round_, feedback)
                print(f"[loop] round {round_} WRITE...", flush=True)
                cand = self.write_candidate(round_, research, feedback)
                # P0-6：契约落盘（WRITE 成功即锚定方向——research/熔断/监督的第一源）
                if not cand.get("parse_failed"):
                    self._write_round_contract(round_, cand, None)
                    self.st.reset_writer_fails()
                else:
                    # P0-4：writer 失败独立计数（模型/协议问题，不烧方向熔断）
                    wf = self.st.bump_writer_fail()
                    self.ev.log_audit("harness", "writer-fail", target=cand["cid"],
                                      round_=round_, detail={"consecutive": wf})
                    if wf >= 3:
                        self.st.update(terminal="pause",
                                       pause={"reason": "writer-fails-3",
                                              "detail": "连续 3 轮 writer 输出无法解析——"
                                                        "模型/协议问题，需人工检查 max_tokens/协议格式"})
                        self.ev.log_audit("harness", "pause", round_=round_,
                                          detail={"reason": "writer-fails-3"})
                        print("[loop] 终局：pause（writer 连续失败）", flush=True)
                        return 0
                print(f"[loop] round {round_} VERIFY {cand['cid']}...", flush=True)
                if cand.get("parse_failed"):
                    vr = {"passed": False, "workloads": [], "error": "write-parse-failed"}
                    self.ev.log_audit("harness", "verify-step", target=cand["cid"], round_=round_,
                                      detail={"rc": 2, "passed": False, "skipped": "parse-failed"})
                else:
                    res = self.exec_stage(round_, cand, "verify")
                    vr = res if res.get("passed") is not None else \
                        {"passed": False, "workloads": res.get("workloads", []),
                         "error": res.get("error", "exec-failed")}
                    # verify 结果也进 solutions.jsonl（CLI 落账逻辑内联——keep P0-1 语义）
                    from harness.core.state import is_pseudo_direction as _ipd
                    direction = cand.get("direction") or "unknown"
                    if vr.get("passed"):
                        self.ev.append_solution(cand["cid"], parent_id=cand.get("parent"),
                                                direction=direction,
                                                hypothesis=str(cand.get("hypothesis", ""))[:200],
                                                status="verified", round_=round_, stage="verify")
                    elif not _ipd(direction):
                        self.ev.append_solution(cand["cid"], parent_id=cand.get("parent"),
                                                direction=direction,
                                                hypothesis=str(cand.get("hypothesis", ""))[:200],
                                                status="reject", round_=round_, stage="verify")
                        n = self.st.bump_direction_fail(str(direction)[:40])
                        self.ev.log_audit("harness", "fuse-check", target=f"direction={direction[:40]}",
                                          round_=round_, detail={"consecutive_fails": n})
                    self.ev.log_audit("harness", "verify-step", target=cand["cid"], round_=round_,
                                      detail={"passed": vr.get("passed"),
                                              "err_ratio": [w.get("err_ratio") for w in vr.get("workloads", [])][:5],
                                              "mode": res.get("verify_mode") if isinstance(res, dict) else None})
                br = None
                if vr.get("passed"):
                    print(f"[loop] round {round_} BENCH...", flush=True)
                    bres = self.exec_stage(round_, cand, "bench")
                    wls = bres.get("workloads", [])
                    mean = sum(w.get("mean_us", 0) for w in wls) / max(len(wls), 1)
                    p50 = sorted(w.get("p50_us", 0) for w in wls)[len(wls) // 2] if wls else None
                    p99 = max((w.get("p99_us", 0) for w in wls), default=None)
                    speedup = (sum(w.get("speedup_vs_ref", 0) for w in wls) / len(wls)
                               if wls and "speedup_vs_ref" in wls[0] else None)
                    import math as _math
                    valid = bool(wls) and _math.isfinite(mean) and mean > 0
                    br = {"mean_us": mean, "p50_us": p50, "p99_us": p99,
                          "speedup": speedup, "valid": valid}
                    # bench 落账（P0-1/P0-5：verdict=benched/invalid）
                    self.ev.append_benchmark(cand["cid"], cand.get("parent"), "P1",
                                             "l0", mean, p50, p99, speedup,
                                             verdict="benched" if valid else "invalid",
                                             note="exec-auto")
                    self.ev.log_audit("harness", "bench-step", target=cand["cid"], round_=round_,
                                      detail={"mean_us": round(mean, 1) if mean else None,
                                              "valid": valid, "workload_set": "l0"})
                print(f"[loop] round {round_} REVIEW...", flush=True)
                verdict = self.review(round_, cand["cid"], vr, br)
                self.st.update(last_verdict=verdict)
                # P0-1：评审终判回写账本（verdict 不再恒 keep——账本=真实状态）
                if not cand.get("parse_failed"):
                    v_final = "keep" if verdict == "COMPLETE" else verdict.lower()
                    self.ev.record_review(cand["cid"], v_final,
                                          note=(self._last_review_text or "")[-200:],
                                          round_=round_)
                    self.ev.log_audit("gate", "review-verdict", target=cand["cid"],
                                      round_=round_, detail={"verdict": v_final})
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
                if verdict == "COMPLETE" and br and br.get("mean_us"):
                    self.mem.add_lesson(round_, "win",
                                        f"{cand.get('direction')} mean={br['mean_us']:.0f}us 刷新最优——该方向有效")
                elif not vr.get("passed"):
                    err = str((vr.get("workloads") or [{}])[0].get("error", ""))[:120]
                    self.mem.add_lesson(round_, "fail",
                                        f"{cand.get('direction')} verify 挂: {err or '数值超差'}——避免同类接口/边界错误")
                # 熔断②接线（v0.2 协议）：REVISE/REJECT 连续 3 次同方向 → 强制换向注入
                # P0-4：伪方向不计数（parse-failed 走 writer_fails 通道）
                if verdict in ("REVISE", "REJECT") and not is_pseudo_direction(cand.get("direction")):
                    n = self.st.bump_direction_fail(str(cand.get("direction", ""))[:40])
                    self.ev.log_audit("harness", "fuse-check", target=f"direction={cand.get('direction', '')[:40]}",
                                      round_=round_, detail={"consecutive_fails": n})
                    if n >= 3:
                        banned = cand.get("direction", "")
                        feedback = (f"FUSE-DIRECTION: 方向「{banned}」已连续 {n} 次未达标，禁止再用。"
                                    f"必须换一个根本不同的优化方向（读 bench 证据找新瓶颈）。")
                        self.ev.log_audit("harness", "fuse", target=f"direction={banned[:40]}",
                                          round_=round_, detail={"reason": "3 consecutive fails", "action": "force-switch"})
                # P0-3：机器漂移判定（每轮 REVIEW 后；不采信 LLM 自评）
                progress = self._drift_check(round_, verdict, br)
                if progress != "ADVANCED" and int(self.st.require().get("stall_count", 0)) >= 3:
                    self.st.update(terminal="stop-drift")
                    self._write_drift_verdict(round_)
                    print("[loop] 终局：stop-drift（连续 3 轮无进展）", flush=True)
                    return 0
                if int(self.st.require().get("stall_count", 0)) == 2:
                    feedback = ("REPLAN: 最近两轮无实质进展（机器漂移判定 STALLED/REGRESSED）。"
                                "读 bench 证据账本找新瓶颈，换根本不同的方向；"
                                "重复同方向的微调不会再有收益。") + ("\n" + feedback if feedback else "")
                    self.ev.log_audit("harness", "replan-inject", round_=round_,
                                      detail={"trigger": "stall_count=2"})
                if verdict in ("COMPLETE", "STOP"):
                    # P1-1：COMPLETE 前 full 档复核（dev 集夺冠不算数——防单形状过拟合）
                    if verdict == "COMPLETE" and br and br.get("valid"):
                        print(f"[loop] round {round_} FULL-SET REBENCH...", flush=True)
                        fres = self.exec_stage(round_, cand, "bench_full")
                        fwls = fres.get("workloads", [])
                        fmean = sum(w.get("mean_us", 0) for w in fwls) / max(len(fwls), 1)
                        import math as _fm
                        fvalid = bool(fwls) and _fm.isfinite(fmean) and fmean > 0
                        if fvalid:
                            self.ev.append_benchmark(cand["cid"], cand.get("parent"), "P1",
                                                     "full", fmean,
                                                     sorted(w.get("p50_us", 0) for w in fwls)[len(fwls) // 2],
                                                     max((w.get("p99_us", 0) for w in fwls), default=None),
                                                     None, verdict="benched", note="full-rebench")
                        best_before = self._best_baseline_us()
                        if not fvalid or (best_before and fmean >= best_before):
                            verdict = "REVISE"
                            self.ev.log_audit("harness", "full-rebench-downgrade",
                                              target=cand["cid"], round_=round_,
                                              detail={"full_mean_us": fmean, "best_before": best_before})
                            feedback = (f"REVISE: full 档复核未达标（full mean="
                                        f"{fmean} vs 最优 {best_before}）"
                                        f"——dev 集结果不可外推，修全形状泛化")
                    if verdict in ("COMPLETE", "STOP"):
                        self.st.update(terminal=verdict)
                        print(f"[loop] 终局：{verdict}", flush=True)
                        return 0
                if verdict == "COMPLETE":
                    subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, capture_output=True)
                    subprocess.run(["git", "commit", "-m",
                                    f"keep({cand['cid']}): auto {cand.get('direction')}"],
                                   cwd=REPO_ROOT, capture_output=True)
            self.st.update(terminal="MAXITER")
            return 0
        except QuotaError as e:
            # P0-7：任何退出路径都有终态（QuotaError→pause 可续）
            self.st.update(terminal="pause", pause={"reason": "quota", "detail": str(e)[:500]})
            self.ev.log_audit("harness", "pause", round_=state.get("round"),
                              detail={"reason": "quota"})
            print(f"[loop] 配额暂停（状态已保存，续跑自动接 round {state.get('round')}）：{e}", flush=True)
            return 0
        except (KeyboardInterrupt, Exception) as e:  # noqa: BLE001
            # P0-7：异常路径写终态 error + 尾部诊断——不再悬空（k1-eager-norm 教训）
            self.st.update(terminal="error", pause={"reason": type(e).__name__,
                                                    "detail": str(e)[:500]})
            self.ev.log_audit("harness", "loop-error", round_=state.get("round"),
                              detail={"type": type(e).__name__, "msg": str(e)[:300]})
            raise

    def _write_drift_verdict(self, round_: int) -> None:
        """P0-3：stop-drift 终态的证据固化（bench 表 + 已试方向——人工复盘入口）。"""
        rows = self.mem.evidence_digest(self._best_baseline_us())
        tried = "\n".join(f"- {s['candidate_id']} [{s.get('status')}] {s.get('direction')}"
                          for s in self.ev.load_solutions()[-12:])
        text = (f"# stop-drift 终局（round {round_}，自动生成）\n\n"
                f"连续 3 轮机器漂移判定非 ADVANCED（数据源 benchmark.csv，不采信 LLM 自评）。\n\n"
                f"## bench 证据\n{rows}\n\n## 已试方向\n{tried}\n")
        (self.task / "docs" / "verdict-drift.md").write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()
    return AutonomousLoop(REPO_ROOT / "tasks" / args.task, args.max_rounds).run()


if __name__ == "__main__":
    raise SystemExit(main())
