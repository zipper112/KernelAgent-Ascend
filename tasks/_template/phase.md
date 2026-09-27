# Phase 任务书：<OP-NAME>（<BATTLE>）

你是 Ascend NPU kernel 优化 agent。本任务目录即你的全部工作区。
**先读同目录 AGENTS.md**（工作流+接口契约+NPU 手册+纪律），再开始。

## 目标

<!-- launcher 注入：主形状口径、基线数字、目标值 -->
TARGET-PLACEHOLDER（例：主形状 w02（B=16 decode）mean_us < <目标>；基线 <数字>）

## 已试方向账本（docs/verdict.md，勿重复）

<!-- launcher 注入：跨战役脉络或"首战，无历史" -->
HISTORY-PLACEHOLDER

## 首推方向

<!-- launcher 注入：来自历史战役/用户提示的技术路线；无则删本节 -->

## 工作节奏（模仿竞赛）

研究（主动查 router/skills）→ 内环 `verify --fast`（几十次是收敛主力）→
全量 verify → `bench.py --record` → 记 solutions.jsonl → 有改进才 git commit → 换向。

## 完成判据

主形状 mean_us < 目标 且其他形状不退化 → solutions.jsonl 标 keep + git commit；
在 docs/verdict.md 写终局报告（结构说明+关键数字+当轮 canonical 输出原文+证据行引用）。
