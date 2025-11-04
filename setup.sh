#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Syncing environment with uv..."
uv sync

echo "==> Activating the environment..."
source .venv/bin/activate

echo "==> Enabling nbdime Git integration..."
nbdime config-git --enable

echo "==> Installing pre-commit hooks..."
pre-commit install
