#!/usr/bin/env python3
"""kda —— KDA-Ascend harness CLI 入口（v0.0 桩）。

v0.0-scaffold：本文件只实现自检类命令（version/status 的一部分），
verify/bench/diagnose/promote/gate 在 Phase 1 落地（见 docs/design/interaction-protocol.md §1）。
桩的行为：未实现命令打印协议文档定位并退出码 2，防止 agent 误判成功。
"""
import argparse
import json
import sys

PROTOCOL_DOC = "docs/design/interaction-protocol.md"
VERSION = "0.0.0-scaffold"

NOT_IMPLEMENTED = ("verify", "bench", "ab", "diagnose", "promote", "gate", "log", "status", "contract", "new-task")


def main() -> int:
    p = argparse.ArgumentParser(prog="kda", description=__doc__)
    p.add_argument("command", choices=["version", *NOT_IMPLEMENTED])
    args, rest = p.parse_known_args()

    if args.command == "version":
        print(json.dumps({"version": VERSION, "protocol": PROTOCOL_DOC}))
        return 0

    print(json.dumps({
        "error": "not-implemented",
        "command": args.command,
        "phase": "Phase 1",
        "spec": f"{PROTOCOL_DOC}#1-cli",
        "note": "v0.0-scaffold 仅含协议与知识资产；harness 代码见路线图",
    }, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    sys.exit(main())
