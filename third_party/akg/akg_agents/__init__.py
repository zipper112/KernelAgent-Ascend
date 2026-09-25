"""akg_agents vendor stub（kda-ascend third_party，ADR-002/008）。

仅保留 KernelVerifier 链所需的包根能力；不拷上游原版 __init__
（其会连带 import core_v2.agents/llm 的 LangGraph 全家桶——vendor 子树不含）。
"""
import os
from pathlib import Path


def get_project_root() -> Path:
    """上游语义（python/akg_agents/__init__.py:54-62）：返回包自身目录。"""
    return Path(os.path.dirname(os.path.abspath(__file__)))


__version__ = "0.0.0-vendor"
