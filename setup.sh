#!/usr/bin/env bash

# setup_uv.sh (to be sourced)
if ( set -euo pipefail
     if ! command -v uv >/dev/null 2>&1; then
       echo "==> Installing uv..."
       curl -LsSf https://astral.sh/uv/install.sh | sh
       export PATH="$HOME/.local/bin:$PATH"
     fi
     echo "==> Syncing environment with uv..."
     uv sync
   ); then
  echo "==> Activating the environment..."
  . .venv/bin/activate
else
  echo "Setup failed"; return 1 2>/dev/null || exit 1
fi
