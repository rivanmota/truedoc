#!/usr/bin/env python3
"""
Offline preflight: is this machine ready to run TrueDoc? Touches no network.

    python3 check_env.py            # backend + web UI
    python3 check_env.py --backend  # backend only (skip the Node checks)

Checks the things that otherwise fail late or, worse, silently:
  * Python version and every backend import,
  * OpenCV's Haar-cascade face detector (OpenCV 5.x has none, and without it
    the portrait signal is quietly switched off and scores shift),
  * Node / npm and the installed web dependencies.

Exit status is 0 when everything needed is present, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys

MIN_PYTHON = (3, 12)
MIN_NODE_MAJOR = 22
HERE = os.path.dirname(os.path.abspath(__file__))

# import name -> pip package (as in requirements.txt)
BACKEND_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "multipart": "python-multipart",
    "requests": "requests",
    "pypdfium2": "pypdfium2",
}
# Only needed to run the test suite.
TEST_MODULES = {"pytest": "pytest", "httpx": "httpx"}

FAILURES: list[str] = []


def ok(msg: str) -> None:
    print(f"  ok       {msg}")


def fail(msg: str, fix: str) -> None:
    print(f"  MISSING  {msg}\n           -> {fix}")
    FAILURES.append(msg)


def warn(msg: str) -> None:
    print(f"  note     {msg}")


def check_python() -> None:
    v = sys.version_info
    if v[:2] >= MIN_PYTHON:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+)",
             "install a newer Python 3 and re-create .venv")


def check_modules(modules: dict[str, str], *, required: bool) -> None:
    for name, package in modules.items():
        try:
            importlib.import_module(name)
        except Exception as exc:  # ImportError, or a native-library load error
            if required:
                fail(f"Python module '{name}' ({type(exc).__name__}: {exc})",
                     f"pip install -r requirements.txt   (package: {package})")
            else:
                warn(f"'{name}' not installed: the test suite cannot run (pip install {package})")
        else:
            ok(f"module {name}")


def check_face_detector() -> None:
    try:
        import cv2
    except Exception:
        return  # already reported by check_modules
    if not hasattr(cv2, "CascadeClassifier"):
        fail(f"OpenCV {cv2.__version__} has no CascadeClassifier (the 5.x wheels dropped it); "
             "portrait analysis would be silently skipped",
             "pip install -r requirements.txt   (pins opencv-python-headless to 4.x)")
        return
    base = getattr(getattr(cv2, "data", None), "haarcascades", None)
    names = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
    paths = [os.path.join(base, n) for n in names] if base else []
    if not paths or not all(os.path.isfile(p) for p in paths):
        fail(f"OpenCV {cv2.__version__} ships no frontal-face Haar cascades", "use the pinned 4.x wheel")
        return
    if any(cv2.CascadeClassifier(p).empty() for p in paths):
        fail("the Haar cascade files did not load", "re-install opencv-python-headless")
        return
    ok(f"OpenCV {cv2.__version__} face cascades load")


def _version_of(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_web() -> None:
    node = shutil.which("node")
    if not node:
        fail("node", "install Node 22+ (https://nodejs.org/), or run the backend only with --backend")
        return
    version = _version_of([node, "--version"]) or ""
    try:
        major = int(version.lstrip("v").split(".")[0])
    except ValueError:
        major = 0
    if major >= MIN_NODE_MAJOR:
        ok(f"Node {version}")
    else:
        fail(f"Node {version or '?'} (need {MIN_NODE_MAJOR}+)", "install a newer Node")
    if shutil.which("npm"):
        ok("npm")
    else:
        fail("npm", "install npm (ships with Node)")
    web = os.path.join(HERE, "web")
    next_bin = os.path.join(web, "node_modules", "next", "package.json")
    if os.path.isfile(next_bin):
        ok("web/node_modules installed")
    else:
        fail("web/node_modules",
             "needs internet once: (cd web && npm ci), or copy a prepared web/node_modules over")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", action="store_true", help="skip the Node / web UI checks")
    args = ap.parse_args()

    print("Backend")
    check_python()
    check_modules(BACKEND_MODULES, required=True)
    check_modules(TEST_MODULES, required=False)
    check_face_detector()
    if not args.backend:
        print("Web UI")
        check_web()

    if FAILURES:
        print(f"\n{len(FAILURES)} problem(s) found. Nothing above needs the network to run once it is installed.")
        return 1
    print("\nReady: everything needed to run TrueDoc is present (no network required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
