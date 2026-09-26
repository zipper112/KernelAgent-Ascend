#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64:/usr/local/Ascend/driver_host/lib64/common:/usr/local/Ascend/driver_host/lib64/driver:$LD_LIBRARY_PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 - <<'PYEOF'
import subprocess
r = subprocess.run(["pip3", "list"], capture_output=True, text=True).stdout
print("ALL triton/torch pkgs:", [l for l in r.splitlines() if "triton" in l.lower() or "torch" in l.lower()])
import importlib.util
for mod in ("triton", "triton_ascend", "triton-ascend"):
    print(mod, "spec:", importlib.util.find_spec(mod) is not None)
PYEOF
