#!/usr/bin/env python3
"""E065 复刻压测（e15）：K8 vec vs orig 双臂 A/B。

口径对齐 npu-7 的 E055/E065 形态：
- PPL 锚点：固定种子文本（E065 验收 |Δ|<0.5%，锚 4.8349——npu-7 长上下文集；e15 无该数据时
  改用本地固定种子的 proxy-PPL：双臂同输入比相对漂移，判据同 |Δ|<0.5%）
- p1 decode：单请求 max_tokens=384（K1 场景 B，测 TPOT）
- p16 prefill+decode：16 并发 × 4k 随机 token × 128 tok（K1 场景 A，测 TTFT/TPOT 尾部）
输出 JSON 行到 stdout，供 kda evidence 记账。
"""
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
ARM = sys.argv[2] if len(sys.argv) > 2 else "vec"
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(resp), time.time() - t0


def wait_ready(timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5)
            return True
        except Exception:
            time.sleep(20)
    return False


def ppl_proxy():
    """固定种子文本的 proxy-PPL：logprob 均值（双臂同输入比相对漂移）。
    vLLM v0.30 口径（实测）：prompt_logprobs 是顶层字段，元素为 {token_id_str: {logprob, rank}}。"""
    random.seed(958)
    ids = [random.randint(100000, 150000) for _ in range(1024)]
    body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 1,
            "prompt_logprobs": 1}
    out, _ = post("/v1/completions", body, timeout=300)
    lps = []
    for c in (out.get("prompt_logprobs") or []):
        if not c:
            continue
        # 每位置字典：取「实际 next token」的 logprob（rank==1 的是 top-1；
        # 实际 token 键即 prompt 的下一个 id——vLLM 把它列在字典里，取非 top-1 判定不可靠，
        # 稳妥口径：取 min(logprob)（最可能的实际 token 是被生成的那个，此处只需双臂一致的标量）
        lps.append(min(v["logprob"] for v in c.values()))
    return statistics.mean(lps) if lps else None


def one_p16(i, results):
    random.seed(958 + i)
    ids = [random.randint(100000, 150000) for _ in range(4096)]
    body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 128,
            "ignore_eos": True}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    c = resp["choices"][0]
    results.append({"i": i, "wall_s": round(dt, 2),
                    "ttft": resp.get("ttft_ms") or c.get("ttft_ms"),
                    "tok": resp.get("usage", {}).get("completion_tokens", 128)})


def bench_p16():
    results = []
    ths = [threading.Thread(target=one_p16, args=(i, results)) for i in range(16)]
    t0 = time.time()
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0
    tps = sum(r["tok"] if isinstance(r.get("tok"), int) else 128 for r in results) / wall
    return {"wall_s": round(wall, 1), "aggregate_tok_s": round(tps, 1),
            "per_req": sorted(results, key=lambda r: r["i"])}


def bench_p1():
    random.seed(958)
    ids = [random.randint(100000, 150000) for _ in range(512)]
    body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 384,
            "ignore_eos": True, "stream": False}
    out, wall = post("/v1/completions", body, timeout=300)
    c = out["choices"][0]
    tok = out.get("usage", {}).get("completion_tokens") or 384
    return {"wall_s": round(wall, 2), "gen_tok": tok,
            "tpot_ms": round(wall * 1000 / max(tok - 1, 1), 1)}


if __name__ == "__main__":
    if not wait_ready():
        print(json.dumps({"error": "serve-not-ready"}))
        sys.exit(2)
    ppl = ppl_proxy()
    p1 = bench_p1()
    p16 = bench_p16()
    print(json.dumps({"arm": ARM, "ppl_proxy": ppl, "p1": p1, "p16_wall": p16["wall_s"],
                      "p16_tok_s": p16["aggregate_tok_s"],
                      "p16_per_req_wall": [r["wall_s"] for r in p16["per_req"]]},
                     ensure_ascii=False))
