"""tests/test_router_unit.py —— knowledge/router/query.py 单元级测试（import 直测）。

期望先行：每用例 docstring 写「给定 → 当 → 则」。
重点：无 PyYAML 回退解析与 yaml 路径**等价**（反编造防线）、expand_symptoms 别名、
score_entry_impl 分支直测、_op_tokens 词元拆分。
"""
import argparse
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "knowledge" / "router"))

import query  # noqa: E402


def _ns(**kw):
    base = dict(symptom=[], op_family=None, arch=None, kind=None, entry=None,
                list_fundamentals=False, production=False, compact=False)
    base.update(kw)
    return argparse.Namespace(**base)


# ---------- load_index 回退解析等价 ----------

def test_load_index_fallback_equivalent_to_yaml(monkeypatch):
    """给定同一 index.yaml → 则【无 PyYAML 回退解析】与【yaml 路径】条目数与每条
    id/symptom/op_family/arch/path→ref 完全一致（反编造防线）。"""
    yaml_entries = query.load_index()          # 正常路径（本机有 PyYAML）
    # 拆掉 yaml：往 sys.modules 塞 None 触发 ImportError
    monkeypatch.setitem(sys.modules, "yaml", None)
    fallback = query.load_index()
    assert len(fallback) == len(yaml_entries) and len(fallback) >= 20
    keys = ("id", "symptom", "op_family", "arch", "path", "ref")
    for a, b in zip(fallback, yaml_entries):
        for k in keys:
            assert a.get(k) == b.get(k), f"fallback!=yaml at {a.get('id')}.{k}: {a.get(k)!r} vs {b.get(k)!r}"


def test_load_index_every_entry_has_id_and_ref():
    """给定索引 → 则每条有 id 且 ref（=path）指向仓内存在的路径。"""
    for e in query.load_index():
        assert e.get("id") and e.get("ref")
        assert (ROOT / e["ref"]).exists(), f"断链 {e['id']} -> {e['ref']}"


# ---------- expand_symptoms ----------

def test_expand_symptoms_alias_and_dedup():
    """给定 ['mem-bound','memory-bound'] → 则归一到 ['memory-bound']（同义词合并，v0.1 语义）。"""
    assert query.expand_symptoms(["mem-bound", "memory-bound"]) == ["memory-bound"]


def test_expand_symptoms_unknown_kept_verbatim():
    """给定未知症状词 → 则原样保留（无法扩展不丢弃）。"""
    out = query.expand_symptoms(["conv-nonexistent"])
    assert out == ["conv-nonexistent"]


def test_expand_symptoms_multiple_words():
    """给定多个词 → 则逐词归一、保序去重。"""
    out = query.expand_symptoms(["mte2", "l2"])
    assert out == ["mte2-bound", "low-l2-hit"]


# ---------- score_entry_impl 直测 ----------

def test_score_arch_filter_rejects():
    """给定 arch=dav_2201 条目 + --arch dav_3510 → 则 None（代际过滤）。"""
    e = {"id": "x", "symptom": ["mte2-bound"], "op_family": ["norm"], "arch": "dav_2201"}
    assert query.score_entry_impl(e, _ns(symptom=["mte2-bound"], op_family="norm", arch="dav_3510")) is None


def test_score_arch_both_passes_any_filter():
    """给定 arch=both 条目 → 则任意 --arch 放行。"""
    e = {"id": "x", "symptom": ["mte2-bound"], "op_family": ["norm"], "arch": "both"}
    assert query.score_entry_impl(e, _ns(symptom=["mte2-bound"], op_family="norm", arch="dav_3510")) is not None


def test_score_kind_prefix_split():
    """给定 kind='guide+template' 条目 + --kind guide → 则放行（前缀语义）。"""
    e = {"id": "x", "symptom": ["*"], "op_family": ["*"], "kind": "guide+template", "arch": "both"}
    assert query.score_entry_impl(e, _ns(kind="guide", op_family="norm")) is not None


def test_score_wildcard_only_is_blindspot():
    """给定仅通配 symptom 命中（无精确 entry）→ 则 None（通配不独立放行=盲区条款）。"""
    e = {"id": "wild", "symptom": ["*"], "op_family": ["norm"], "arch": "both"}
    assert query.score_entry_impl(e, _ns(symptom=["mte2-bound"], op_family="norm")) is None


def test_score_wildcard_with_exact_entry_passes():
    """给定通配条目 + --entry 精确取 → 则放行（id 查询路径不受盲区条款限制）。"""
    e = {"id": "wild", "symptom": ["*"], "op_family": ["norm"], "arch": "both"}
    assert query.score_entry_impl(e, _ns(entry="wild")) is not None


def test_score_fundamental_requires_entry_or_list():
    """给定无症状轴的 fundamental 条目 → 则普通 symptom 查询 None、--entry 放行、
    --list-fundamentals 放行。"""
    e = {"id": "fund-norm", "symptom": [], "op_family": ["norm"], "kind": "fundamental", "arch": "both"}
    assert query.score_entry_impl(e, _ns(symptom=["mte2-bound"])) is None
    assert query.score_entry_impl(e, _ns(entry="fund-norm")) is not None
    assert query.score_entry_impl(e, _ns(list_fundamentals=True)) is not None


def test_score_exact_family_beats_wildcard_family():
    """给定精确族条目与通配族条目 → 则精确族得分更高（5 > 3）。"""
    exact = {"id": "a", "symptom": ["mte2-bound"], "op_family": ["norm"], "arch": "both"}
    wild = {"id": "b", "symptom": ["mte2-bound"], "op_family": ["*"], "arch": "both"}
    s1 = query.score_entry_impl(exact, _ns(symptom=["mte2-bound"], op_family="norm"))
    s2 = query.score_entry_impl(wild, _ns(symptom=["mte2-bound"], op_family="norm"))
    assert s1 > s2 > 0


# ---------- _op_tokens（--production 词元匹配核心） ----------

def test_op_tokens_split_underscore_and_versions():
    """给定真实命名风格 → 则：rms_norm→{rms,norm}；add_v2→{add}（尾部版本剥除）；
    RMSNorm→{rmsnorm}（驼峰整体小写）。"""
    assert query._op_tokens("rms_norm") == {"rms", "norm"}
    assert query._op_tokens("add_v2") == {"add"}
    assert query._op_tokens("RMSNorm") == {"rmsnorm"}


def test_op_tokens_substring_not_token():
    """给定 'random_normal' 与 family='norm' → 则 norm ∉ 词元（子串不算，杜绝误入）。"""
    assert "norm" not in query._op_tokens("random_normal")


def test_op_tokens_legacy_nv2_kept_intact():
    """给定 accumulate_nv2（历史名，非版本后缀）→ 则词元含 nv2 而非脏 'n'。"""
    assert query._op_tokens("accumulate_nv2") == {"accumulate", "nv2"}


def test_op_tokens_empty_parts_dropped():
    """给定 'a__b' → 则空段丢弃；裸 'v123' 是词元非版本（新版法语义）→ 保留。"""
    assert query._op_tokens("a__b") == {"a", "b"}
    assert query._op_tokens("v123") == {"v123"}
    assert query._op_tokens("x_v2") == {"x"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
