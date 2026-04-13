from __future__ import annotations

import os
import sys
import traceback


def main() -> None:
    print("=== Python diagnostics ===")
    print("sys.executable:", sys.executable)
    print("sys.version:", sys.version.replace("\n", " "))
    print("cwd:", os.getcwd())
    print("sys.path[0:5]:")
    for p in sys.path[:5]:
        print(" -", p)

    print("\n=== Torch import diagnostics ===")
    try:
        import torch  # noqa: F401

        import torch as _torch

        print("torch imported OK")
        print("torch.__version__:", _torch.__version__)
        print("torch.version.cuda:", getattr(_torch.version, "cuda", None))
        print("cuda available:", _torch.cuda.is_available())
    except Exception as exc:
        print("torch import FAILED:", repr(exc))
        traceback.print_exc()


if __name__ == "__main__":
    main()

