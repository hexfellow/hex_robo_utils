#!/usr/bin/env bash
set -Eeuo pipefail
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-28
################################################################

# Parse command line arguments
MINIMAL=false
TEST=false
while [[ $# -gt 0 ]]; do
  case $1 in
  --min)
    MINIMAL=true
    shift
    ;;
  --test)
    TEST=true
    shift
    ;;
  *)
    echo "Unknown option: $1"
    echo "Usage: $0 [--min]"
    echo "  --min : Install with minimal package"
    exit 1
    ;;
  esac
done

CUR_DIR="$(pwd)"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "CUR_DIR: $CUR_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"

cd $SCRIPT_DIR

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found. Please install uv first." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11
fi
source .venv/bin/activate

# Install hex_robo_utils
rm -rf dist build *.egg-info
uv pip uninstall hex_robo_utils || true

if [ "$MINIMAL" = true ]; then
  echo "Installing minimal package..."
  uv pip install -e .
else
  echo "Installing with [all] extras..."
  uv pip install -e .[all]
fi

if [ "$TEST" = true ]; then
  echo "Running tests..."
  uv pip install pytest
fi

cd $CUR_DIR
