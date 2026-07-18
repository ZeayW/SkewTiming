#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${TOOLS_DIR:?set TOOLS_DIR}"
: "${DATASET_DIR:?set DATASET_DIR}"

exec "$PYTHON_BIN" "$TOOLS_DIR/inspect_serialized_dataset.py" "$DATASET_DIR"
