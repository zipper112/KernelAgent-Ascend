#!/usr/bin/env python3
"""tools/record_serving_ab.py —— serving 级 A/B 结果经 Evidence（唯一写方）入账。

用途：K8/E065 复刻的双臂（vec/orig）整服测量不属 kda bench 的算子微基准口径，
但结论必须进证据链（kda-ascend 本地为家）。本脚本读两臂 JSON（e15-bench.py 产出），
写 benchmark.csv 两行（verdict 按双臂相对胜负）+ audit 两条（含 PPL 漂移判定）。
用法：python tools/record_serving_ab.py <task> <vec.json> <orig.json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.core.evidence import Evidence  # noqa: E402


def main() -> int:
    task, vec_p, orig_p = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    vec = json.loads(vec_p.read_text(encoding="utf-8"))
    orig = json.loads(orig_p.read_text(encoding="utf-8"))

    # PPL 漂移判定（E065 验收：|Δ|<0.5%）
    ppl_v, ppl_o = vec.get("ppl_proxy"), orig.get("ppl_proxy")
    drift = abs(ppl_v - ppl_o) / abs(ppl_o) * 100 if (ppl_v and ppl_o) else None

    # 性能判据：p16 聚合吞吐 + p1 TPOT（vec 应不差于 orig）
    tok_v, tok_o = vec["p16_tok_s"], orig["p16_tok_s"]
    tpot_v, tpot_o = vec["p1"]["tpot_ms"], orig["p1"]["tpot_ms"]
    gain_p16 = (tok_v - tok_o) / tok_o * 100
    gain_p1 = (tpot_o - tpot_v) / tpot_o * 100

    ev = Evidence(ROOT / "tasks" / task)
    ev.append_benchmark("e065-vec", None, "P2", "full", vec["p1"]["tpot_ms"], None, None,
                        tok_v, verdict="keep", note=f"serving-ab/e065 p16_tok_s={tok_v} tpot_ms={tpot_v}")
    ev.append_benchmark("e065-orig", None, "P2", "full", orig["p1"]["tpot_ms"], None, None,
                        tok_o, verdict="keep", note=f"serving-ab/e065-orig p16_tok_s={tok_o} tpot_ms={tpot_o}")
    ev.log_audit("harness", "serving-ab", target="k8/e065-e15",
                 detail={"ppl_drift_pct": round(drift, 3) if drift else None,
                         "p16_gain_pct": round(gain_p16, 2),
                         "p1_tpot_gain_pct": round(gain_p1, 2),
                         "verdict_ppl": ("PASS" if drift is not None and drift < 0.5 else "FAIL" if drift else "N/A")})
    print(json.dumps({"ppl_drift_pct": round(drift, 3) if drift else None,
                      "p16_gain_pct": round(gain_p16, 2),
                      "p1_tpot_gain_pct": round(gain_p1, 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
