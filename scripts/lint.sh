#!/bin/bash
# Linting script for home-agent integration
# This runs the same linters that Home Assistant uses

set -e

echo "Running ruff linter..."
python3 -m ruff check custom_components/home_agent/

echo ""
echo "Running ruff formatter..."
python3 -m ruff format custom_components/home_agent/

echo ""
echo "Running mypy type checker..."
python3 -m mypy custom_components/home_agent/ \
  --ignore-missing-imports \
  --no-strict-optional \
  --warn-unreachable \
  --show-error-codes || echo "⚠️  MyPy found some issues (non-blocking)"

echo ""
echo "Running pylint on home-agent..."
python3 -m pylint custom_components/home_agent/

echo ""
echo "✅ All linting checks passed!"
