#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_DATASETS:?set SOURCE_DATASETS to colon-separated packaged datasets}"
: "${MERGED_DATASET:?set MERGED_DATASET to a new local directory}"
: "${DESTINATION_DIR:?set DESTINATION_DIR on the destination host}"

TOOLS_DIR=${TOOLS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
EXPECTED_CASES=${EXPECTED_CASES:-1}

if [[ -e "${MERGED_DATASET}" ]]; then
    echo "refusing to overwrite existing merged dataset: ${MERGED_DATASET}" >&2
    exit 1
fi

mkdir -p "${MERGED_DATASET}"
IFS=: read -r -a source_datasets <<< "${SOURCE_DATASETS}"

for source_dataset in "${source_datasets[@]}"; do
    if [[ ! -d "${source_dataset}" ]]; then
        echo "source dataset does not exist: ${source_dataset}" >&2
        exit 1
    fi

    source_name=$(basename "$(dirname "${source_dataset}")")
    if [[ -f "${source_dataset}/manifest.json" ]]; then
        cp -p \
            "${source_dataset}/manifest.json" \
            "${MERGED_DATASET}/${source_name}.manifest.json"
    fi

    found_design=0
    for design_dir in "${source_dataset}"/*/; do
        [[ -d "${design_dir}" ]] || continue
        design=$(basename "${design_dir}")
        if [[ -e "${MERGED_DATASET}/${design}" ]]; then
            echo "duplicate design in source datasets: ${design}" >&2
            exit 1
        fi
        cp -a "${design_dir}" "${MERGED_DATASET}/${design}"
        found_design=1
    done
    if [[ "${found_design}" -eq 0 ]]; then
        echo "no design directories in source dataset: ${source_dataset}" >&2
        exit 1
    fi
done

python3 "${TOOLS_DIR}/validate_dataset.py" \
    --dataset-dir "${MERGED_DATASET}" \
    --expected-cases "${EXPECTED_CASES}"

SOURCE_DIR="${MERGED_DATASET}" \
DESTINATION_DIR="${DESTINATION_DIR}" \
DESTINATION_HOST="${DESTINATION_HOST:-zywang@projgw}" \
DESTINATION_PORT="${DESTINATION_PORT:-2349}" \
    "${TOOLS_DIR}/transfer_dataset.sh"
