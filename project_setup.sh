#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# -----------------------------------------------------------------------------
# Project bootstrap script (uv version)
# -----------------------------------------------------------------------------
# This script bootstraps the development environment using the `uv` project and
# package manager from Astral. It will:
#   1. Install `uv` if it isn't already available on the system.
#   2. Create (or reuse) a local `.venv` virtual environment managed by `uv`.
#   3. Resolve and lock dependencies from `pyproject.toml` → `uv.lock`.
#   4. Sync (install) all dependencies into the virtual environment.
#   5. Register a dedicated Jupyter kernel associated with the environment.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 1. Ensure `uv` is available
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] 'uv' not found – installing …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer typically puts the binary in ~/.cargo/bin or ~/.local/bin
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

echo "[bootstrap] Using uv version: $(uv --version)"

# -----------------------------------------------------------------------------
# 2. Create or reuse the virtual environment (managed by uv)
# -----------------------------------------------------------------------------
if [ ! -d .venv ]; then
    echo "[bootstrap] Creating virtual environment (.venv) with uv"
    uv venv
fi

export UV_HTTP_TIMEOUT=180

# -----------------------------------------------------------------------------
# 3. Resolve & lock dependencies, then install them
# -----------------------------------------------------------------------------
# Generate / update the cross-platform lockfile (uv.lock)
uv lock

# Install (sync) all dependencies into the virtual environment
uv sync

# -----------------------------------------------------------------------------
# 4. Register a Jupyter kernel bound to this environment (optional)
# -----------------------------------------------------------------------------
source .venv/bin/activate
python -m ipykernel install --user --name=thinkchain --display-name "ThinkChain"

echo "[bootstrap] Jupyter kernel 'thinkchain' installed."

echo "[bootstrap] Project setup complete – environment managed by uv."