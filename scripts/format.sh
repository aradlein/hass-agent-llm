#!/bin/bash
# Quick formatting script for home-agent integration

set -e

echo "Running ruff auto-fix..."
python3 -m ruff check --fix custom_components/home_agent/

echo ""
echo "Running ruff formatter..."
python3 -m ruff format custom_components/home_agent/

echo ""
echo "✅ Code formatted!"
