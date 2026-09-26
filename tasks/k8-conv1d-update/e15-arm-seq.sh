#!/bin/bash
# 单臂完整序列：等就绪 → 预热 p16 一轮（丢弃）→ 正式测（PPL+p1+p16）
# 用法：bash e15-arm-seq.sh <tag>    产出 /data02/kda/<tag>.json
TAG=${1:?need-tag}
cd /data02/kda
python3 - <<PYEOF
import json, time, urllib.request, threading, random, statistics
import importlib.util
spec = importlib.util.spec_from_file_location("b", "/data02/kda/e15-bench.py")
b = importlib.util.module_from_spec(spec)
import sys
sys.argv = ["e15-bench.py", "8001", "warmup"]
spec.loader.exec_module(b)          # exec 到 wait_ready 为止有副作用？不——直接用函数
PYEOF
echo "fallback-to-python-driver"
python3 - <<'PYEOF'
import json, time, urllib.request, threading, random, statistics, sys
PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"

def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read()), time.time() - t0

def wait_ready(timeout_s=2400):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5); return True
        except Exception: time.sleep(20)
    return False

def p16_round(tag, seed_base):
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
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0
    return {"wall_s": round(wall, 1), "tok_s": round(sum(r["tok"] for r in results) / wall, 1),
            "per_req": [r["wall_s"] for r in sorted(results, key=lambda r: r["i"])]}

def p1_round():
    random.seed(958)
    ids = [random.randint(100000, 150000) for _ in range(512)]
    out, wall = post("/v1/completions", {"model": "GLM-5.3-Flash", "prompt": ids,
                       "max_tokens": 384, "ignore_eos": True}, timeout=300)
    tok = out.get("usage", {}).get("completion_tokens") or 384
    return {"wall_s": round(wall, 2), "tpot_ms": round(wall * 1000 / max(tok - 1, 1), 1)}

assert wait_ready(), "serve not ready"
p16_round("warm", 1)                       # 预热轮（丢弃）
p1 = p1_round()                            # p1 也预热一次
p1 = p1_round()
r1 = p16_round("正式1", 100)
r2 = p16_round("正式2", 200)
print(json.dumps({"arm": sys.argv[1] if len(sys.argv) > 1 else "?",
                  "p1": p1, "p16_r1": r1, "p16_r2": r2}))
PYEOF
