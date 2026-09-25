"""python -m harness.cli 入口兜底（Windows Device Guard 拦截 pip console script 时的等价通道）。"""
import sys

from harness.cli import main

if __name__ == "__main__":
    sys.exit(main())
