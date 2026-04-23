#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _editable_spec(*, include_dev: bool, include_notebooks: bool) -> str:
    extras: list[str] = []
    if include_dev:
        extras.append("dev")
    if include_notebooks:
        extras.append("notebooks")
    if not extras:
        return "."
    return f".[{','.join(extras)}]"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap an editable development environment for this repository."
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Install only runtime dependencies and the editable package.",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Do not upgrade pip/setuptools/wheel before installing the package.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    editable_spec = _editable_spec(
        include_dev=not args.runtime_only,
        include_notebooks=not args.runtime_only,
    )

    if not args.skip_pip_upgrade:
        _run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools>=61", "wheel"],
            cwd=repo_root,
        )

    _run([sys.executable, "-m", "pip", "install", "-e", editable_spec], cwd=repo_root)

    print()
    print("Bootstrap complete.")
    print(f"Installed editable target: {editable_spec}")
    print(f"Repository root: {repo_root}")
    print(f"Python executable: {sys.executable}")


if __name__ == "__main__":
    main()
