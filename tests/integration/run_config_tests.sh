#!/bin/bash
# Script to run configuration variation tests with proper markers
#
# This script demonstrates how to run the new configuration variation tests.
# It uses pytest markers to control which tests run based on available services.

set -e

echo "=================================="
echo "Configuration Variation Test Suite"
echo "=================================="
echo ""

# Check if services are available (optional)
if command -v docker &> /dev/null; then
    echo "✓ Docker available"
else
    echo "⚠ Docker not available - some tests may skip"
fi

echo ""
echo "Running tests..."
echo ""

# Run with verbose output and show which tests are skipped/passed
pytest tests/integration/test_config_variations.py \
    -v \
    --tb=short \
    --color=yes \
    -p no:warnings

echo ""
echo "=================================="
echo "Test Summary"
echo "=================================="
echo ""
echo "The tests verify the following configuration options:"
echo ""
echo "1. LLM Backends:"
echo "   - llama-cpp, vllm-server, ollama-gpu"
echo "   - Verifies X-Ollama-Backend header is set correctly"
echo ""
echo "2. Context Formats:"
echo "   - natural_language, hybrid"
echo "   - Verifies output format matches configuration"
echo ""
echo "3. Embedding Providers:"
echo "   - openai"
echo "   - Verifies correct API path is used"
echo ""
echo "4. Memory Extraction LLM:"
echo "   - local"
echo "   - Verifies correct LLM is used for memory extraction"
echo ""
echo "See CONFIG_VARIATION_COVERAGE.md for detailed documentation."
echo ""
