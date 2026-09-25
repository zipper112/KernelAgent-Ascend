"""tests/test_akg_vendor.py —— vendored akg 子树可导入性断言（自审批次 B3）。"""
from conftest import AKG_IMPORTABLE, needs_akg


@needs_akg
def test_kernel_verifier_importable():
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier
    assert KernelVerifier.__name__ == "KernelVerifier"


@needs_akg
def test_register_local_worker_importable():
    from akg_agents.core.worker.manager import register_local_worker
    assert callable(register_local_worker)


@needs_akg
def test_stub_root():
    import akg_agents
    assert akg_agents.get_project_root().name == "akg_agents"


@needs_akg
def test_jinja_templates_vendored():
    root = __import__("akg_agents").get_project_root()
    t = root / "op" / "resources" / "templates"
    assert (t / "kernel_verify_template_refactored.j2").exists()
    assert (root / "utils" / "compile_tools" / "ascend_compile" / "run.sh").exists()


def test_vendor_layout_documented():
    """布局与 ADR/维护文档一致（third_party/akg/akg_agents 包根）。"""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "third_party" / "akg" / "akg_agents" / "op" / "verifier" / "kernel_verifier.py").exists()
    assert (root / "third_party" / "akg" / "akg_agents" / "examples" / "run_kernel_profile.py").exists()
