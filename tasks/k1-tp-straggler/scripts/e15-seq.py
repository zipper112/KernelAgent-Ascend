#!/usr/bin/env python3
"""K1 三臂稳态序列（v2：p16 加流式 TTFT——chunk 边界 straggler 的客户端显形）。
序列：等就绪 → 预热 p16（丢弃）→ p1×2 → p16 r1（全 miss 主判据）→ p16 r2。
TTFT = 流式首个非空 text chunk 的到达时间。"""
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
TAG = sys.argv[1] if len(sys.argv) > 1 else "?"


def wait_ready(timeout_s=2400):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5)
            return True
        except Exception:
            time.sleep(20)
    return False


def one_p16(i, seed_base, out):
    random.seed(seed_base + i)
    ids = [random.randint(100000, 150000) for _ in range(4096)]
    body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 128,
            "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ttft, tok = None, None
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            ch = d.get("choices")
            if ttft is None and ch and (ch[0].get("text") or ch[0].get("token_ids") is not None):
                ttft = (time.time() - t0) * 1000
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                tok = u["completion_tokens"]
        wall = time.time() - t0
        out.append({"i": i, "wall_s": round(wall, 2),
                    "ttft_ms": round(ttft, 1) if ttft else None,
                    "tok": tok or 128, "stream_ok": True})
    except Exception as e:  # 流式失败 → 非流式兜底（丢 TTFT 保吞吐口径）
        body.pop("stream"); body.pop("stream_options")
        req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out.append({"i": i, "wall_s": round(time.time() - t1, 2), "ttft_ms": None,
                    "tok": d.get("usage", {}).get("completion_tokens", 128),
                    "stream_ok": False, "err": str(e)[:60]})


def p16_round(seed_base):
    out = []
    ths = [threading.Thread(target=one_p16, args=(i, seed_base, out)) for i in range(16)]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    per = sorted(r["wall_s"] for r in out)
    tt = sorted(r["ttft_ms"] for r in out if r["ttft_ms"])
    p99 = tt[-1] if tt else None
    p50 = statistics.median(tt) if tt else None
    return {"wall_s": round(wall, 1),
            "tok_s": round(sum(r["tok"] for r in out) / wall, 1),
            "req_wall_p50": round(statistics.median(per), 2),
            "ttft_p50_ms": round(p50, 1) if p50 else None,
            "ttft_p99_ms": round(p99, 1) if p99 else None,
            "ttft_max_ms": round(max(tt), 1) if tt else None,
            "stream_fail": sum(1 for r in out if not r["stream_ok"])}


def p1_round():
    random.seed(958)
    ids = [random.randint(100000, 150000) for _ in range(512)]
    body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 384,
            "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    wall = time.time() - t0
    tok = d.get("usage", {}).get("completion_tokens") or 384
    return {"wall_s": round(wall, 2), "tpot_ms": round(wall * 1000 / max(tok - 1, 1), 1)}


if __name__ == "__main__":
    assert wait_ready(), "serve not ready"
    p16_round(1)            # 预热（丢弃）
    p1_round()              # p1 预热（丢弃）
    p1a, p1b = p1_round(), p1_round()
    r1 = p16_round(100)     # 全 miss 主判据
    r2 = p16_round(200)
    print(json.dumps({"arm": TAG, "p1": [p1a, p1b], "p16_r1": r1, "p16_r2": r2},
                     ensure_ascii=False))
