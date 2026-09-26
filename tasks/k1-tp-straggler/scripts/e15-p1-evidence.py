#!/usr/bin/env python3
"""K1 取证驱动：开 profiler → p1 单流 decode（K1 场景 B）→ 关 profiler。
产出：/prof 下 torch trace（容器内）→ 宿主 /data02/kda/prof/。"""
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8001"


def post(path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout).read()
    return r[:200], time.time() - t0


def wait_ready(timeout_s=2400):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5)
            return True
        except Exception:
            time.sleep(20)
    return False


if __name__ == "__main__":
    assert wait_ready(), "serve not ready"
    # 预热（graph 换档等一次性成本先烧掉，别污染取证窗）
    random.seed(1)
    ids = [random.randint(100000, 150000) for _ in range(512)]
    post("/v1/completions", {"model": "GLM-5.3-Flash", "prompt": ids,
                             "max_tokens": 64, "ignore_eos": True})
    time.sleep(3)
    # 开采样
    print("start_profile:", post("/start_profile", {"output_dir": "/prof"}, timeout=120)[0][:80])
    # 取证负载：p1 单流 max_tokens 384（K1 场景 B 复刻）
    random.seed(958)
    ids = [random.randint(100000, 150000) for _ in range(512)]
    _, wall = post("/v1/completions", {"model": "GLM-5.3-Flash", "prompt": ids,
                                       "max_tokens": 384, "ignore_eos": True})
    print(f"p1 load done in {wall:.1f}s")
    time.sleep(2)
    print("stop_profile:", post("/stop_profile", {}, timeout=300)[0][:80])
    time.sleep(5)     # 落盘
    print("EVIDENCE-DONE")
