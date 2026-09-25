"""tests/conftest.py —— 共享 fixture / 路径。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "akg"))   # vendored akg_agents（ADR-002/008/B3）

AKG_IMPORTABLE = True
try:
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier  # noqa: F401,E402
except Exception:  # noqa: BLE001 —— jinja2 缺失等环境差异
    AKG_IMPORTABLE = False

needs_akg = __import__("pytest").mark.skipif(
    not AKG_IMPORTABLE, reason="vendored akg_agents 不可 import（装 jinja2；LocalWorker 链还需 pandas/torch_npu——上板机项）")
