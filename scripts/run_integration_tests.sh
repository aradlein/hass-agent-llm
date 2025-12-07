#!/bin/bash

# Integration Test Runner
# This script runs integration tests that require external services (ChromaDB, LLM, etc.)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Integration Test Runner"
echo "=================================================="
echo ""

# Load environment variables from .env.test
ENV_FILE="$PROJECT_ROOT/.env.test"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: .env.test file not found!${NC}"
    echo "Please copy .env.test.example to .env.test and configure it with your settings."
    echo ""
    echo "  cp .env.test.example .env.test"
    echo "  # Edit .env.test with your configuration"
    echo ""
    exit 1
fi

echo -e "${GREEN}Loading configuration from .env.test...${NC}"
set -a
source "$ENV_FILE"
set +a
echo ""

# Verify required environment variables
REQUIRED_VARS=(
    "TEST_CHROMADB_HOST"
    "TEST_CHROMADB_PORT"
    "TEST_LLM_BASE_URL"
    "TEST_LLM_API_KEY"
    "TEST_EMBEDDING_BASE_URL"
    "TEST_EMBEDDING_API_KEY"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}Error: Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    exit 1
fi

# Health check functions
check_chromadb() {
    echo -n "Checking ChromaDB connectivity... "
    # Try v2 API first (newer), then fall back to v1 (legacy)
    local url_v2="http://${TEST_CHROMADB_HOST}:${TEST_CHROMADB_PORT}/api/v2/heartbeat"
    local url_v1="http://${TEST_CHROMADB_HOST}:${TEST_CHROMADB_PORT}/api/v1/heartbeat"

    if curl -s -f -m 5 "$url_v2" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC} (v2 API)"
        return 0
    elif curl -s -f -m 5 "$url_v1" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC} (v1 API)"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Tried: $url_v2"
        echo "  Tried: $url_v1"
        return 1
    fi
}

check_llm_endpoint() {
    echo -n "Checking LLM endpoint connectivity... "
    local url="${TEST_LLM_BASE_URL}/models"

    if curl -s -f -m 5 -H "Authorization: Bearer ${TEST_LLM_API_KEY}" "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  URL: $url"
        return 1
    fi
}

check_embedding_endpoint() {
    echo -n "Checking embedding endpoint connectivity... "
    # Try Ollama-style endpoint first, then OpenAI-compatible
    local url_ollama="${TEST_EMBEDDING_BASE_URL}/api/tags"
    local url_openai="${TEST_EMBEDDING_BASE_URL}/v1/models"
    local url_models="${TEST_EMBEDDING_BASE_URL}/models"

    if curl -s -f -m 5 "$url_ollama" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC} (Ollama)"
        return 0
    elif curl -s -f -m 5 -H "Authorization: Bearer ${TEST_EMBEDDING_API_KEY}" "$url_openai" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC} (OpenAI)"
        return 0
    elif curl -s -f -m 5 -H "Authorization: Bearer ${TEST_EMBEDDING_API_KEY}" "$url_models" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Tried: $url_ollama"
        echo "  Tried: $url_openai"
        echo "  Tried: $url_models"
        return 1
    fi
}

# Run health checks
echo "=================================================="
echo "Service Health Checks"
echo "=================================================="
echo ""

HEALTH_CHECK_FAILED=0

check_chromadb || HEALTH_CHECK_FAILED=1
check_llm_endpoint || HEALTH_CHECK_FAILED=1
check_embedding_endpoint || HEALTH_CHECK_FAILED=1

echo ""

if [ $HEALTH_CHECK_FAILED -eq 1 ]; then
    echo -e "${RED}Health checks failed. Please ensure all services are running.${NC}"
    echo ""
    exit 1
fi

# Run integration tests
echo "=================================================="
echo "Running Integration Tests"
echo "=================================================="
echo ""

cd "$PROJECT_ROOT"

# Parse command line arguments
# Run all tests in the integration folder by default (not just marked ones)
PYTEST_ARGS="tests/integration/ -v -n 2"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            PYTEST_ARGS="tests/integration/ -m smoke -v"
            shift
            ;;
        --performance)
            PYTEST_ARGS="tests/integration/ -m performance -v"
            shift
            ;;
        --marked-only)
            PYTEST_ARGS="tests/integration/ -m integration -v"
            shift
            ;;
        -k)
            EXTRA_ARGS="$EXTRA_ARGS -k $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Build allow-hosts list from configured endpoints
# Extract hostnames from configured URLs and allow them for socket connections
ALLOW_HOSTS="127.0.0.1,::1"

# Add ChromaDB host
if [ -n "$TEST_CHROMADB_HOST" ]; then
    ALLOW_HOSTS="$ALLOW_HOSTS,$TEST_CHROMADB_HOST"
fi

# Add LLM host (extract from URL)
if [ -n "$TEST_LLM_BASE_URL" ]; then
    LLM_HOST=$(echo "$TEST_LLM_BASE_URL" | sed -E 's|https?://([^:/]+).*|\1|')
    ALLOW_HOSTS="$ALLOW_HOSTS,$LLM_HOST"
fi

# Add embedding host (extract from URL)
if [ -n "$TEST_EMBEDDING_BASE_URL" ]; then
    EMB_HOST=$(echo "$TEST_EMBEDDING_BASE_URL" | sed -E 's|https?://([^:/]+).*|\1|')
    ALLOW_HOSTS="$ALLOW_HOSTS,$EMB_HOST"
fi

echo "Allowed hosts for network connections: $ALLOW_HOSTS"
echo ""

# Run pytest with HA plugin disabled (which blocks sockets for integration tests)
# The -p no:homeassistant option disables pytest-homeassistant-custom-component's socket blocking
# Add -ra flag to show summary of all test results (passed, failed, skipped, etc.)
# Suppress RuntimeWarning for background memory extraction tasks (these are intentional "fire and forget" operations)
PYTEST_WARNINGS="-W ignore::RuntimeWarning"
echo "Running: pytest $PYTEST_ARGS -p no:homeassistant -ra $PYTEST_WARNINGS $EXTRA_ARGS"
echo ""

if pytest $PYTEST_ARGS -p no:homeassistant -ra $PYTEST_WARNINGS $EXTRA_ARGS; then
    echo ""
    echo -e "${GREEN}=================================================="
    echo "Integration Tests PASSED"
    echo "==================================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}=================================================="
    echo "Integration Tests FAILED"
    echo "==================================================${NC}"
    exit 1
fi
