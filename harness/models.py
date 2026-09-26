"""harness/models.py —— 模型调用唯一入口（ADR-004/006；协议 §7）。

- 读 agent-config/models.yaml（角色→模型映射）+ infra/secrets/provider（key 唯一来源）；
- OpenAI Chat Completion 兼容协议（urllib 直发，零第三方依赖）；
- usage 两级记账（run-global + 任务级，Evidence.append_usage）——额度消耗从此可见；
- 契约：网络类错误重试 ≤ max_retries；402/429 抛 QuotaError（调用方走 checkpoint-pause，绝不重试）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness.core.evidence import Evidence

REPO_ROOT = Path(__file__).resolve().parent.parent


class QuotaError(RuntimeError):
    """402/429：配额耗尽/限流。调用方必须 checkpoint-pause（协议 §7.1），绝不重试。"""


class ModelsClient:
    def __init__(self, task_root: Path, models_yaml: Path | None = None):
        import yaml
        self.cfg = yaml.safe_load((models_yaml or REPO_ROOT / "agent-config" / "models.yaml")
                                  .read_text(encoding="utf-8"))
        self.task_root = Path(task_root)
        self.ev = Evidence(self.task_root)
        self.global_ledger = REPO_ROOT / "run-global" / "usage.jsonl"
        self._providers = self.cfg.get("providers", {})
        self._defaults = self.cfg.get("defaults", {})

    # ---------- 角色解析 ----------

    def _role_cfg(self, role: str) -> dict:
        r = self.cfg["roles"][role]
        return {"model": r["primary"]["model"],
                "base_url": self._providers[r["primary"]["provider"]]["base_url"],
                "fallbacks": [(f["model"], self._providers[f["provider"]]["base_url"])
                              for f in r.get("fallbacks", [])],
                "env_key": r["primary"].get("env_key", "GLM_API_KEY")}

    def _key(self) -> str:
        from infra.secrets.provider import get_glm_key
        ref = get_glm_key()
        if not ref:
            raise RuntimeError("无 GLM key（GLM_API_KEY 或 local-secrets.yaml）")
        return ref.get()

    # ---------- 调用 ----------

    def chat(self, role: str, messages: list[dict], temperature: float = 0.3,
             max_tokens: int | None = None, purpose: str = "",
             thinking: str | None = "disabled") -> str:
        """角色调用：返回 assistant 文本。usage 记账（两级）；QuotaError 上抛。
        max_tokens 缺省 = models.yaml 的 max_tokens_per_call（轮级上限，硬执行）。
        thinking：GLM-5.3 是推理模型——不关思考时 16k 预算会被 reasoning_content
        吃光（finish_reason=length 且 content 空，实测 2026-09-26）。循环自带
        评审与迭代反馈，模型内部慢思考冗余 → 默认 disabled；需要时显式传 'enabled'。"""
        rc = self._role_cfg(role)
        cap = max_tokens or int(self._defaults.get("max_tokens_per_call", 8192))
        self._thinking = thinking
        key = self._key()
        attempts = [(rc["model"], rc["base_url"]), *rc["fallbacks"]]
        max_retries = int(self._defaults.get("max_retries", 2))
        last_err: Exception | None = None
        for mi, (model, base_url) in enumerate(attempts):
            for retry in range(max_retries + 1):
                try:
                    return self._call_once(model, base_url, key, messages,
                                           temperature, cap, role, purpose)
                except QuotaError:
                    raise                      # 402/429：不重试不换档，直接上抛 checkpoint
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                    last_err = e
                    if retry < max_retries:
                        time.sleep(2 * (retry + 1))
            # 本档重试耗尽 → 换 fallback 档（auto_rules ③）
        raise RuntimeError(f"所有端点失败：{last_err}")

    def _call_once(self, model, base_url, key, messages, temperature, max_tokens,
                   role, purpose) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        th = getattr(self, "_thinking", "disabled")
        if th == "disabled":
            body["thinking"] = {"type": "disabled"}   # GLM 推理模型：不关则预算被思维链吃光
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        timeout = int(self._defaults.get("timeout_s", 300))
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (402, 429):
                raise QuotaError(f"HTTP {e.code}: {e.read()[:200]!r}")
            raise
        data = json.loads(resp.read())
        usage = data.get("usage", {}) or {}
        p_tok = int(usage.get("prompt_tokens", 0))
        c_tok = int(usage.get("completion_tokens", 0))
        self.ev.append_usage(task=self.task_root.name, role=role, model=model,
                             prompt_tokens=p_tok, completion_tokens=c_tok,
                             source="api", global_ledger=self.global_ledger)
        if purpose:
            self.ev.log_audit("harness", "llm-call", target=f"{role}:{model}",
                              detail={"purpose": purpose, "prompt_tokens": p_tok,
                                      "completion_tokens": c_tok})
        choices = data.get("choices") or [{}]
        msg = (choices[0].get("message") or {})
        content = msg.get("content")
        finish = choices[0].get("finish_reason")
        if not content:
            # 计费但空内容（实测 16k 输出时出现）：抛诊断而非静默空串烧循环
            raise RuntimeError(
                f"empty content (finish_reason={finish}, billed c={c_tok}); "
                f"reasoning_content head={str(msg.get('reasoning_content'))[:120]!r}")
        return content
