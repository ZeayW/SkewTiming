#!/usr/bin/env bash
set -euo pipefail

: "${TOOLS_DIR:?set TOOLS_DIR}"
: "${CASE_DIRS:?set colon-separated CASE_DIRS}"
: "${REFERENCE_GOLDEN:?set REFERENCE_GOLDEN}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${DESIGN_NAME:?set DESIGN_NAME}"

IFS=: read -r -a case_dirs <<< "$CASE_DIRS"
args=()
for case_dir in "${case_dirs[@]}"; do
  args+=(--case-dir "$case_dir")
done

python3 "$TOOLS_DIR/recover_report_dataset.py" \
  "${args[@]}" \
  --reference-golden "$REFERENCE_GOLDEN" \
  --output-dir "$OUTPUT_DIR" \
  --design-name "$DESIGN_NAME"
