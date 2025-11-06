. setup.sh

echo "==> Enabling nbdime Git integration..."
nbdime config-git --enable

echo "==> Installing pre-commit hooks..."
pre-commit install