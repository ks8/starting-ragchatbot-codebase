#!/bin/bash
# Run code quality checks for the project

set -e

echo "=== Code Quality Checks ==="
echo ""

# Determine mode: "check" (default) or "fix"
MODE="${1:-check}"

if [ "$MODE" = "fix" ]; then
    echo "--- Running black (auto-format) ---"
    uv run black backend/ main.py
    echo ""
    echo "All formatting fixes applied."
elif [ "$MODE" = "check" ]; then
    echo "--- Running black (check only) ---"
    uv run black --check backend/ main.py
    echo ""
    echo "All checks passed."
else
    echo "Usage: $0 [check|fix]"
    echo "  check  - Verify formatting (default)"
    echo "  fix    - Auto-fix formatting issues"
    exit 1
fi
