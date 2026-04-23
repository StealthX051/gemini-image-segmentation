# Setup

This document is the canonical setup and environment guide for a fresh clone.

## Goal
- A fresh clone should become development-ready on Windows, Linux, or macOS with the fewest manual steps possible.
- The recommended workflows below install the package itself, its runtime dependencies, test tooling, and notebook tooling.

## Recommended Path: Conda

```bash
conda env create -f environment.yml
conda run -n gemini_seg python scripts/bootstrap_env.py
```

Why this is the safest default:
- it gives a consistent Python version (`3.11`)
- it installs the repo itself in editable mode with the `dev` and `notebooks` extras
- it avoids relying on shell-specific `conda activate` behavior
- it provides the Cairo runtime used by `cairosvg`

## Alternative Path: venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap_env.py
```

This path is useful on Linux or macOS hosts without Conda.
On Debian/Ubuntu-like systems, install `python3-venv` first if `python3 -m venv` is unavailable:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

## Linux Note For CairoSVG
The repo uses `cairosvg` for Moondream SVG mask rasterization. On Linux, a pure `venv` install may require system Cairo libraries. If `cairosvg` fails to import, install the Cairo runtime for your distro and rerun the bootstrap step.

For Debian/Ubuntu-like systems, common packages are:

```bash
sudo apt-get update
sudo apt-get install -y libcairo2 libffi-dev
```

## What The Bootstrap Script Does
- upgrades `pip`, `setuptools`, and `wheel`
- installs the repo in editable mode
- installs the `dev` and `notebooks` extras by default
- keeps NumPy on the `1.x` line for compatibility with the compiled scientific stack currently used in this repo

Use the bootstrap script when:
- you are using the `venv` path
- you need to refresh an existing Conda environment after dependency metadata changes
- you want a runtime-only editable install

Runtime-only install:

```bash
python scripts/bootstrap_env.py --runtime-only
```

Manual equivalent:

```bash
python -m pip install -e .[dev,notebooks]
```

Requirements-file wrapper:

```bash
python -m pip install -r requirements-dev.txt
```

## Verification
After setup, verify the environment with a few fast checks:

```bash
python -c "import gemini_segmentation, nanobanana_segmentation; print('package import ok')"
gemini-seg --help
nanobanana-study --help
python -m gemini_segmentation.cli --help
pytest -q tests/test_config.py tests/test_prompts.py tests/test_batch.py
```

If you plan to use the NanoBanana service lane:

```bash
python -c "from fastapi.testclient import TestClient; print('service deps ok')"
```

## Repo Hygiene
- Local-only environments and caches such as `.venv/`, `venv/`, `.tmp-bootstrap-venv/`, `.pytest_cache/`, and `__pycache__/` are intentionally gitignored.
- Text file line endings are standardized for portability:
  - Python, shell, Markdown, YAML, TOML, JSON, and notebook files use LF
  - PowerShell scripts use CRLF
- If you create a scratch environment inside the repo for a one-off bootstrap smoke test, keep it in one of the ignored directories above or delete it before sharing the worktree.

## Environment Variables
Create a `.env` file with the API keys you need:
- `GOOGLE_API_KEY`
- `MOONDREAM_API_KEY`
- `REPLICATE_API_TOKEN`

The direct CLI does not auto-load `.env`, so export it in the active shell before direct CLI calls.

```bash
set -a
source .env
set +a
```

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $n, $v = $_ -split '=', 2
  Set-Item -Path "Env:$n" -Value $v.Trim().Trim('"').Trim("'")
}
```

## Entry Points After Install
These console commands are available after editable install:
- `gemini-seg`
- `nanobanana-study`

The module entrypoints remain valid too:
- `python -m gemini_segmentation.cli`
- `python -m nanobanana_segmentation.study.runner`
