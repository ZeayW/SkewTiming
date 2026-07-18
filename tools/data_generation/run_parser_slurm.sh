#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${PARSER_SRC:?set PARSER_SRC}"
: "${RAW_DATA_DIR:?set RAW_DATA_DIR}"
: "${DATASET_DIR:?set DATASET_DIR}"

cd "$PARSER_SRC"
exec "$PYTHON_BIN" parser.py \
  --rawdata_path "$RAW_DATA_DIR" \
  --data_savepath "$DATASET_DIR" \
  --min_cases_per_design "${MIN_CASES_PER_DESIGN:-1}" \
  --parser_workers "${PARSER_WORKERS:-1}" \
  --log_level "${LOG_LEVEL:-0}"
