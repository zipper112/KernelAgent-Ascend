import json

o = json.load(open("/data02/kda/orig-seq.json"))
v = json.load(open("/data02/kda/vec-seq.json"))
for name, d in [("ORIG", o), ("VEC ", v)]:
    print(name, "p1_tpot:", [x["tpot_ms"] for x in d["p1"]],
          "p16_tok_s:", [d["p16_r1"]["tok_s"], d["p16_r2"]["tok_s"]],
          "p16_p50:", [d["p16_r1"]["p50"], d["p16_r2"]["p50"]])
# 稳态口径：p1 取第二次；p16 取 r1（首轮全 miss 口径，两臂同种子 → 同前缀命中条件）
p1o = o["p1"][1]["tpot_ms"]
p1v = v["p1"][1]["tpot_ms"]
p16o = o["p16_r1"]["tok_s"]
p16v = v["p16_r1"]["tok_s"]
print(f"判定（p1 第2次 / p16 r1 全 miss）：p1 {p1o}->{p1v} ms ({(p1v-p1o)/p1o*100:+.1f}%)；p16 {p16o}->{p16v} tok/s ({(p16v-p16o)/p16o*100:+.1f}%)")
