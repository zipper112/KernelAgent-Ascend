#!/usr/bin/env python3
"""单臂完整序列驱动：等就绪 → 预热 p16 → p1×2 → p16×2（稳态口径）。
用法：python3 e15-seq.py <tag>；产出 JSON 打到 stdout。"""
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


def post(body, timeout=600):
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read()), time.time() - t0


def wait_ready(timeout_s=2400):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5)
            return True
        except Exception:
            time.sleep(20)
    return False


def p16_round(seed_base):
    results = []

    def one(i):
        random.seed(seed_base + i)
        ids = [random.randint(100000, 150000) for _ in range(4096)]
        body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 128, "ignore_eos": True}
        t0 = time.time()
        req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
        results.append({"i": i, "wall_s": round(time.time() - t0, 2),
                        "tok": resp.get("usage", {}).get("completion_tokens", 128)})

    ths = [threading.Thread(target=one, args=(i,)) for i in range(16)]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    per = [r["wall_s"] for r in sorted(results, key=lambda r: r["i"])]
    return {"wall_s": round(wall, 1), "tok_s": round(sum(r["tok"] for r in results) / wall, 1),
            "per_req": per, "p50": round(statistics.median(per), 2)}


def p1_round():
    random.seed(958)
    ids = [random.randint(100000, 150000) for _ in range(512)]
    out, wall = post({"model": "GLM-5.3-Flash", "prompt": ids,
                      "max_tokens": 384, "ignore_eos": True}, timeout=300)
    tok = out.get("usage", {}).get("completion_tokens") or 384
    return {"wall_s": round(wall, 2), "tpot_ms": round(wall * 1000 / max(tok - 1, 1), 1)}


if __name__ == "__main__":
    assert wait_ready(), "serve not ready"
    p16_round(1)            # 预热（丢弃）
    p1_round()              # p1 预热（丢弃）
    p1a, p1b = p1_round(), p1_round()
    r1, r2 = p16_round(100), p16_round(200)
    print(json.dumps({"arm": TAG, "p1": [p1a, p1b],
                      "p16_r1": r1, "p16_r2": r2}, ensure_ascii=False))
