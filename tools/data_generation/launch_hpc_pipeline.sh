#!/usr/bin/env bash
set -euo pipefail

: "${HPC_HOST:?set HPC_HOST to hpc1, ..., hpc8}"
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
SCRATCH_PARENT=${SCRATCH_PARENT:-/tmp}

case "${HPC_HOST}" in
    hpc[1-8]) ;;
    *)
        echo "HPC_HOST must be one of hpc1, ..., hpc8" >&2
        exit 2
        ;;
esac

ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${HPC_HOST}" \
    bash -s -- \
    "${TD_SOURCE}" \
    "${TOOLS_SOURCE}" \
    "${RTL_SOURCE}" \
    "${DESIGN}" \
    "${RESULT_ROOT}" \
    "${SDC_COUNT}" \
    "${STA_WORKERS}" \
    "${PATH_NUM}" \
    "${ENDPOINT_NUM}" \
    "${TD_THREADS}" \
    "${READ_CHECKPOINT}" \
    "${QOR_MONITOR}" \
    "${GENERATE_ONLY}" \
    "${ALLOW_BLACK_BOXES}" \
    "${SCRATCH_PARENT}" <<'REMOTE'
set -euo pipefail

TD_SOURCE=$1
TOOLS_SOURCE=$2
RTL_SOURCE=$3
DESIGN=$4
RESULT_ROOT=$5
SDC_COUNT=$6
STA_WORKERS=$7
PATH_NUM=$8
ENDPOINT_NUM=$9
TD_THREADS=${10}
READ_CHECKPOINT=${11}
QOR_MONITOR=${12}
GENERATE_ONLY=${13}
ALLOW_BLACK_BOXES=${14}
SCRATCH_PARENT=${15}

if [[ -e "${RESULT_ROOT}" ]]; then
    echo "result directory already exists: ${RESULT_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}"

nohup env \
    TD_SOURCE="${TD_SOURCE}" \
    TOOLS_SOURCE="${TOOLS_SOURCE}" \
    RTL_SOURCE="${RTL_SOURCE}" \
    DESIGN="${DESIGN}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    SDC_COUNT="${SDC_COUNT}" \
    STA_WORKERS="${STA_WORKERS}" \
    PATH_NUM="${PATH_NUM}" \
    ENDPOINT_NUM="${ENDPOINT_NUM}" \
    TD_THREADS="${TD_THREADS}" \
    READ_CHECKPOINT="${READ_CHECKPOINT}" \
    QOR_MONITOR="${QOR_MONITOR}" \
    GENERATE_ONLY="${GENERATE_ONLY}" \
    ALLOW_BLACK_BOXES="${ALLOW_BLACK_BOXES}" \
    SCRATCH_PARENT="${SCRATCH_PARENT}" \
    "${TOOLS_SOURCE}/run_slurm_pipeline.sh" \
    > "${RESULT_ROOT}/hpc_pipeline.log" 2>&1 < /dev/null &
pid=$!

{
    echo "host=$(hostname)"
    echo "pid=${pid}"
    echo "launched_at=$(date --iso-8601=seconds)"
} > "${RESULT_ROOT}/hpc_launcher.txt"

echo "host=$(hostname) pid=${pid} result_root=${RESULT_ROOT}"
REMOTE
