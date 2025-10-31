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
echo "Running pylint on home-agent..."
python3 -m pylint custom_components/home_agent/

echo ""
echo "✅ All linting checks passed!"
