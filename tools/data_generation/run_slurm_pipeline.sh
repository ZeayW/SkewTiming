#!/usr/bin/env bash
set -euo pipefail

: "${TD_SOURCE:?set TD_SOURCE to the TangDynasty release directory}"
: "${TOOLS_SOURCE:?set TOOLS_SOURCE to this tools directory}"
: "${RTL_SOURCE:?set RTL_SOURCE to one self-contained Verilog file}"
: "${DESIGN:?set DESIGN to a filesystem-safe design name}"
: "${RESULT_ROOT:?set RESULT_ROOT to a unique shared result directory}"

SDC_COUNT=${SDC_COUNT:-1}
STA_WORKERS=${STA_WORKERS:-1}
PATH_NUM=${PATH_NUM:-1}
ENDPOINT_NUM=${ENDPOINT_NUM:-200000}
TD_THREADS=${TD_THREADS:-auto}
READ_CHECKPOINT=${READ_CHECKPOINT:-1}
QOR_MONITOR=${QOR_MONITOR:-0}
GENERATE_ONLY=${GENERATE_ONLY:-0}
ALLOW_BLACK_BOXES=${ALLOW_BLACK_BOXES:-0}
JOB_TOKEN=${SLURM_JOB_ID:-manual_$$}
SCRATCH_PARENT=${SCRATCH_PARENT:-${SLURM_TMPDIR:-/tmp}}
SCRATCH_ROOT=${SCRATCH_PARENT}/nuatimer_${JOB_TOKEN}_${DESIGN}
LOCAL_TD=${SCRATCH_ROOT}/td
LOCAL_TOOLS=${SCRATCH_ROOT}/tools
LOCAL_RTL=${SCRATCH_ROOT}/rtl/${DESIGN}.v
LOCAL_WORK=${SCRATCH_ROOT}/work
LOCAL_DATASET=${SCRATCH_ROOT}/dataset
AUDIT_DIR=${RESULT_ROOT}/audit
COMPAT_LIB_SOURCE=${COMPAT_LIB_SOURCE:-${TOOLS_SOURCE}/compat_lib}

archive_results() {
    exit_code=$?
    mkdir -p "${AUDIT_DIR}"
    printf '%s\n' "${exit_code}" > "${AUDIT_DIR}/exit_code.txt"
    if [[ -d "${LOCAL_WORK}" ]]; then
        find "${LOCAL_WORK}" -maxdepth 1 -type f -exec cp -p {} "${AUDIT_DIR}/" \;
        if [[ -d "${LOCAL_WORK}/projects" ]]; then
            find "${LOCAL_WORK}/projects" -mindepth 1 -maxdepth 2 -type f \
                -printf '%s %p\n' | sort -k2 > "${RESULT_ROOT}/generated_inventory.txt"
            find "${LOCAL_WORK}/projects" -maxdepth 1 -type f -exec cp -p {} "${AUDIT_DIR}/" \;
            find "${LOCAL_WORK}/projects" -mindepth 2 -maxdepth 2 -type f \
                \( -name 'nua_label.*' -o -name 'golden_labeled.txt' \) \
                -exec cp -p {} "${AUDIT_DIR}/" \;
        fi
    fi
    if [[ -d "${LOCAL_DATASET}" && "${exit_code}" -eq 0 ]]; then
        mkdir -p "${RESULT_ROOT}/dataset"
        cp -a "${LOCAL_DATASET}/." "${RESULT_ROOT}/dataset/"
    fi
    rm -rf "${SCRATCH_ROOT}"
    exit "${exit_code}"
}
trap archive_results EXIT

mkdir -p "${LOCAL_TD}" "${LOCAL_TOOLS}" "$(dirname "${LOCAL_RTL}")" "${RESULT_ROOT}"

# Loading the release into node-local storage avoids paging the commercial
# executable and architecture database through the login-node NFS mount.
cp -R \
    "${TD_SOURCE}/bin" \
    "${TD_SOURCE}/lib" \
    "${TD_SOURCE}/arch" \
    "${TD_SOURCE}/ip" \
    "${TD_SOURCE}/license" \
    "${TD_SOURCE}/packages" \
    "${TD_SOURCE}/pubkey" \
    "${LOCAL_TD}/"
cp -R "${TOOLS_SOURCE}/." "${LOCAL_TOOLS}/"
cp "${RTL_SOURCE}" "${LOCAL_RTL}"
if [[ -d "${COMPAT_LIB_SOURCE}" ]]; then
    cp -R "${COMPAT_LIB_SOURCE}/." "${LOCAL_TD}/lib/"
fi

{
    echo "job_id=${SLURM_JOB_ID:-none}"
    echo "hostname=$(hostname)"
    echo "design=${DESIGN}"
    echo "sdc_count=${SDC_COUNT}"
    echo "sta_workers=${STA_WORKERS}"
    echo "path_num=${PATH_NUM}"
    echo "endpoint_num=${ENDPOINT_NUM}"
    echo "td_threads=${TD_THREADS}"
    echo "read_checkpoint=${READ_CHECKPOINT}"
    echo "qor_monitor=${QOR_MONITOR}"
    echo "scratch_parent=${SCRATCH_PARENT}"
    echo "generate_only=${GENERATE_ONLY}"
    echo "allow_black_boxes=${ALLOW_BLACK_BOXES}"
    echo "rtl_source=${RTL_SOURCE}"
    echo "rtl_sha256=$(sha256sum "${RTL_SOURCE}" | awk '{print $1}')"
    echo "started_at=$(date --iso-8601=seconds)"
} > "${RESULT_ROOT}/job_metadata.txt"

generate_args=(
    --td-root "${LOCAL_TD}"
    --rtl "${LOCAL_RTL}"
    --design-name "${DESIGN}"
    --work-dir "${LOCAL_WORK}"
    --sdcs "${SDC_COUNT}"
    --execute
)
if [[ "${ALLOW_BLACK_BOXES}" == "1" ]]; then
    generate_args+=(--allow-black-boxes)
fi

python3 "${LOCAL_TOOLS}/generate_cases.py" "${generate_args[@]}"

find "${LOCAL_WORK}/projects" -mindepth 1 -maxdepth 2 -type f \
    -printf '%s %p\n' | sort -k2 > "${RESULT_ROOT}/generated_inventory.txt"

if [[ "${GENERATE_ONLY}" == "1" ]]; then
    echo "completed_at=$(date --iso-8601=seconds)" >> "${RESULT_ROOT}/job_metadata.txt"
    exit 0
fi

sta_args=(
    --td-root "${LOCAL_TD}"
    --projects-dir "${LOCAL_WORK}/projects"
    --workers "${STA_WORKERS}"
    --path-num "${PATH_NUM}"
    --endpoint-num "${ENDPOINT_NUM}"
    --td-threads "${TD_THREADS}"
)
if [[ "${READ_CHECKPOINT}" == "0" ]]; then
    sta_args+=(--no-read-checkpoint)
fi
if [[ "${QOR_MONITOR}" == "1" ]]; then
    sta_args+=(--qor-monitor)
fi

python3 "${LOCAL_TOOLS}/run_sta.py" "${sta_args[@]}"

python3 "${LOCAL_TOOLS}/package_dataset.py" \
    --projects-dir "${LOCAL_WORK}/projects" \
    --output-dir "${LOCAL_DATASET}"

python3 "${LOCAL_TOOLS}/validate_dataset.py" \
    --dataset-dir "${LOCAL_DATASET}" \
    --expected-cases "${SDC_COUNT}"

echo "completed_at=$(date --iso-8601=seconds)" >> "${RESULT_ROOT}/job_metadata.txt"
