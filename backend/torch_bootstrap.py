from __future__ import annotations

import os
import sys


def prepare_torch_environment() -> None:
    if sys.platform != "win32" or sys.version_info < (3, 8):
        return

    dll_candidates = [
        *[os.path.join(path, "torch", "lib") for path in sys.path],
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Packages",
            "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0",
            "LocalCache",
            "local-packages",
            "Python311",
            "site-packages",
            "torch",
            "lib",
        ),
    ]

    for candidate in dll_candidates:
        if os.path.exists(candidate):
            try:
                os.add_dll_directory(candidate)
            except Exception:
                pass
