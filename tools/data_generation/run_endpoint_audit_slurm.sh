#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${TOOLS_DIR:?set TOOLS_DIR}"
: "${RAW_DATA_DIR:?set RAW_DATA_DIR}"
: "${DATASET_DIR:?set DATASET_DIR}"

exec "$PYTHON_BIN" "$TOOLS_DIR/audit_endpoint_retention.py" \
  --raw-data "$RAW_DATA_DIR" \
  --dataset "$DATASET_DIR"
