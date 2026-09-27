# Phase 任务书：K11 index-fused（K9 档承接）

先读 AGENTS.md。

## 目标

w02（T=4096 prefill chunk）mean_us 显著优于 c000 基线（上游 5 连发串原样包装）；
w01/w03 不退化。起步目标 -30%，K10 同病根已拿 -70% 参考。

## c000 基线

solution/c000/candidate.py = 上游等价串：x.index_select(0,idx_spec)+index_select(0,idx_ns)
+ zeros 装配（reference 的 input_select 部分）+ index_copy_ 回写。先 bench --record 建行。

## 已试方向

首战无历史。K10 教训：包装税省完后看访存模式（本任务访存=两段 gather+一段写，
理论下界≈3×T×D×2B）。

## 完成判据

w02 < c000×0.7 且全形状不退化 → keep + commit + verdict.md（附当轮 canonical 原文）。
