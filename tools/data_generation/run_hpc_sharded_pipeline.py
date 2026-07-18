#!/usr/bin/env python3
"""Generate one design once and distribute STA cases across HPC hosts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from common import atomic_write_json, sha256_file, validate_labeled_golden


SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=60",
    "-o",
    "ServerAliveCountMax=10",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
)


FRONTEND_SCRIPT = r"""
set -euo pipefail

TD_SOURCE=$1
TOOLS_SOURCE=$2
RTL_SOURCE=$3
DESIGN=$4
CASE_COUNT=$5
RESULT_ROOT=$6
TOKEN=$7
SCRATCH_PARENT=$8

PROJECTS_OUT=${RESULT_ROOT}/generated/projects
if [[ -f "${RESULT_ROOT}/generated/frontend.complete" ]]; then
    echo "frontend already complete"
    exit 0
fi

LOCAL_ROOT=${SCRATCH_PARENT}/nuatimer_${TOKEN}_frontend
LOCAL_TD=${LOCAL_ROOT}/td
LOCAL_TOOLS=${LOCAL_ROOT}/tools
LOCAL_WORK=${LOCAL_ROOT}/work
LOCAL_RTL=${LOCAL_ROOT}/rtl/${DESIGN}.v

rm -rf "${LOCAL_ROOT}"
mkdir -p "${LOCAL_TD}" "${LOCAL_TOOLS}" "$(dirname "${LOCAL_RTL}")"
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

python3 "${LOCAL_TOOLS}/generate_cases.py" \
    --td-root "${LOCAL_TD}" \
    --rtl "${LOCAL_RTL}" \
    --design-name "${DESIGN}" \
    --work-dir "${LOCAL_WORK}" \
    --sdcs "${CASE_COUNT}" \
    --execute \
    --allow-black-boxes

# NUIAT changes the SDC and golden file, not the generated netlist. Validate
# normalized content before replacing duplicate 100+ MB files with hard links.
python3 - "${LOCAL_TOOLS}" "${LOCAL_WORK}/projects" "${DESIGN}" "${CASE_COUNT}" <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from common import sha256_td_netlist_content

projects = Path(sys.argv[2])
design = sys.argv[3]
case_count = int(sys.argv[4])
netlists = [
    projects / f"{design}_{index}" / f"{design}_{index}.v"
    for index in range(case_count)
]
hashes = {sha256_td_netlist_content(path) for path in netlists}
if len(hashes) != 1:
    raise RuntimeError(f"generated cases contain {len(hashes)} distinct netlists")
canonical = netlists[0]
for netlist in netlists[1:]:
    netlist.unlink()
    os.link(canonical, netlist)
print(f"deduplicated {len(netlists)} netlists: {next(iter(hashes))}")
PY

mkdir -p "${RESULT_ROOT}/generated"
rm -rf "${PROJECTS_OUT}"
cp -a "${LOCAL_WORK}/projects" "${PROJECTS_OUT}"
cp -p "${LOCAL_WORK}/generation_status.json" "${RESULT_ROOT}/generated/"
cp -p "${LOCAL_WORK}/generation.stdout.log" "${RESULT_ROOT}/generated/"
cp -p "${LOCAL_WORK}/generation.stderr.log" "${RESULT_ROOT}/generated/"
find "${PROJECTS_OUT}" -mindepth 1 -maxdepth 2 -type f \
    -printf '%i %s %p\n' | sort -k3 > "${RESULT_ROOT}/generated/inventory.txt"
date --iso-8601=seconds > "${RESULT_ROOT}/generated/frontend.complete"
rm -rf "${LOCAL_ROOT}"
"""


SHARD_SCRIPT = r"""
set -euo pipefail

TD_SOURCE=$1
TOOLS_SOURCE=$2
DESIGN=$3
RESULT_ROOT=$4
TOKEN=$5
HOST_NAME=$6
CASES=$7
WORKERS=$8
PATH_NUM=$9
ENDPOINT_NUM=${10}
TD_THREADS=${11}
ATTEMPT=${12}
SCRATCH_PARENT=${13}

LOCAL_ROOT=${SCRATCH_PARENT}/nuatimer_${TOKEN}_${HOST_NAME}
LOCAL_TD=${LOCAL_ROOT}/td
LOCAL_TOOLS=${LOCAL_ROOT}/tools
LOCAL_PROJECTS=${LOCAL_ROOT}/projects
SHARD_OUT=${RESULT_ROOT}/shards/${HOST_NAME}/attempt_${ATTEMPT}
mkdir -p "${SHARD_OUT}"

if [[ ! -f "${LOCAL_ROOT}/stage.complete" ]]; then
    rm -rf "${LOCAL_ROOT}"
    mkdir -p "${LOCAL_TD}" "${LOCAL_TOOLS}" "${LOCAL_ROOT}"
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
    if [[ -d "${TOOLS_SOURCE}/compat_lib" ]]; then
        cp -R "${TOOLS_SOURCE}/compat_lib/." "${LOCAL_TD}/lib/"
    fi
    cp -R --preserve=links --no-preserve=mode,ownership,timestamps \
        "${RESULT_ROOT}/generated/projects" "${LOCAL_PROJECTS}"
    date --iso-8601=seconds > "${LOCAL_ROOT}/stage.complete"
fi

set +e
python3 "${LOCAL_TOOLS}/run_sta.py" \
    --td-root "${LOCAL_TD}" \
    --projects-dir "${LOCAL_PROJECTS}" \
    --design "${DESIGN}" \
    --cases "${CASES}" \
    --workers "${WORKERS}" \
    --path-num "${PATH_NUM}" \
    --endpoint-num "${ENDPOINT_NUM}" \
    --td-threads "${TD_THREADS}" \
    > "${SHARD_OUT}/run_sta.stdout.log" \
    2> "${SHARD_OUT}/run_sta.stderr.log"
RUN_STATUS=$?
set -e

if [[ -f "${LOCAL_PROJECTS}/nua_label.summary.json" ]]; then
    cp -p "${LOCAL_PROJECTS}/nua_label.summary.json" "${SHARD_OUT}/"
fi
IFS=',' read -r -a CASE_INDEXES <<< "${CASES}"
for index in "${CASE_INDEXES[@]}"; do
    case_name=${DESIGN}_${index}
    local_case=${LOCAL_PROJECTS}/${case_name}
    shared_case=${RESULT_ROOT}/generated/projects/${case_name}
    mkdir -p "${SHARD_OUT}/${case_name}"
    for name in golden_labeled.txt nua_label.status.json nua_label.stdout.log nua_label.stderr.log; do
        if [[ -f "${local_case}/${name}" ]]; then
            cp -p "${local_case}/${name}" "${SHARD_OUT}/${case_name}/"
        fi
    done
    if [[ -f "${local_case}/golden_labeled.txt" ]]; then
        cp -p "${local_case}/golden_labeled.txt" "${shared_case}/golden_labeled.txt.tmp"
        mv "${shared_case}/golden_labeled.txt.tmp" "${shared_case}/golden_labeled.txt"
    fi
    if [[ -f "${local_case}/nua_label.status.json" ]]; then
        cp -p "${local_case}/nua_label.status.json" "${shared_case}/nua_label.status.json.tmp"
        mv "${shared_case}/nua_label.status.json.tmp" "${shared_case}/nua_label.status.json"
    fi
done
printf '%s\n' "${RUN_STATUS}" > "${SHARD_OUT}/exit_code.txt"
date --iso-8601=seconds > "${SHARD_OUT}/completed_at.txt"
exit "${RUN_STATUS}"
"""


PACKAGE_SCRIPT = r"""
set -euo pipefail

TOOLS_SOURCE=$1
DESIGN=$2
CASE_COUNT=$3
RESULT_ROOT=$4
TOKEN=$5
SCRATCH_PARENT=$6

LOCAL_ROOT=${SCRATCH_PARENT}/nuatimer_${TOKEN}_package
rm -rf "${LOCAL_ROOT}"
mkdir -p "${LOCAL_ROOT}/tools"
cp -R "${TOOLS_SOURCE}/." "${LOCAL_ROOT}/tools/"
cp -a "${RESULT_ROOT}/generated/projects" "${LOCAL_ROOT}/projects"
python3 "${LOCAL_ROOT}/tools/package_dataset.py" \
    --projects-dir "${LOCAL_ROOT}/projects" \
    --output-dir "${LOCAL_ROOT}/dataset" \
    --design "${DESIGN}"
python3 "${LOCAL_ROOT}/tools/validate_dataset.py" \
    --dataset-dir "${LOCAL_ROOT}/dataset" \
    --expected-cases "${CASE_COUNT}"
rm -rf "${RESULT_ROOT}/dataset"
cp -a "${LOCAL_ROOT}/dataset" "${RESULT_ROOT}/dataset"
date --iso-8601=seconds > "${RESULT_ROOT}/dataset.complete"
rm -rf "${LOCAL_ROOT}"
"""


CLEANUP_SCRIPT = r"""
set -euo pipefail
TOKEN=$1
SCRATCH_PARENT=$2
HOST_NAME=$3
rm -rf "${SCRATCH_PARENT}/nuatimer_${TOKEN}_${HOST_NAME}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-root", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--rtl", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=100)
    parser.add_argument("--hosts", default="hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8")
    parser.add_argument("--frontend-host", default="hpc8")
    parser.add_argument("--package-host", default="hpc8")
    parser.add_argument("--path-num", type=int, default=1)
    parser.add_argument("--endpoint-num", type=int, default=200000)
    parser.add_argument("--td-threads", default="auto")
    parser.add_argument("--scratch-parent", default="/dev/shm")
    parser.add_argument(
        "--attempt-worker-caps",
        default="0,4,1",
        help="per-host worker caps; 0 means all assigned cases",
    )
    return parser.parse_args()


def log(message: str, log_path: Path) -> None:
    line = "{} {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_remote(host: str, script: str, arguments: Sequence[object], output: Path) -> int:
    command = ["ssh", *SSH_OPTIONS, host, "bash", "-s", "--"]
    command.extend(str(item) for item in arguments)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            input=script,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def distribute(indices: Sequence[int], hosts: Sequence[str]) -> Dict[str, List[int]]:
    result = {host: [] for host in hosts}
    for offset, index in enumerate(indices):
        result[hosts[offset % len(hosts)]].append(index)
    return {host: values for host, values in result.items() if values}


def missing_cases(projects: Path, design: str, case_count: int) -> List[int]:
    missing = []
    for index in range(case_count):
        case_dir = projects / "{}_{}".format(design, index)
        label = case_dir / "golden_labeled.txt"
        status_path = case_dir / "nua_label.status.json"
        try:
            validation = validate_labeled_golden(
                label, require_nuiat_startpoints=True
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") != "complete":
                raise ValueError("case status is not complete")
            if status.get("requested_nuiat_startpoint_count") != validation["pi_count"]:
                raise ValueError("requested NUIAT startpoint count differs")
            if status.get("unexpected_startpoint_path_count") != 0:
                raise ValueError("unexpected timing startpoint paths were reported")
            if status.get("unexpected_startpoint_count") != 0:
                raise ValueError("unexpected timing startpoints were reported")
            if status.get("timing_path_count") != status.get(
                "nuiat_startpoint_path_count"
            ):
                raise ValueError("not every timing path starts at a NUIAT PI")
            if status.get("endpoint_count") != validation["endpoint_count"]:
                raise ValueError("status endpoint count differs")
            if status.get("label_count") != validation["label_count"]:
                raise ValueError("status label count differs")
        except Exception:
            missing.append(index)
    return missing


def write_status(path: Path, state: str, **values: object) -> None:
    atomic_write_json(path, {"state": state, "updated_at": time.time(), **values})


def require_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> int:
    args = parse_args()
    if args.case_count < 1:
        raise ValueError("--case-count must be positive")
    hosts = [item.strip() for item in args.hosts.split(",") if item.strip()]
    if not hosts or any(host not in {"hpc{}".format(i) for i in range(1, 9)} for host in hosts):
        raise ValueError("--hosts must contain hpc1 through hpc8")
    if args.frontend_host not in hosts or args.package_host not in hosts:
        raise ValueError("frontend and package hosts must be included in --hosts")
    caps = [int(item) for item in args.attempt_worker_caps.split(",")]
    if not caps or any(cap < 0 for cap in caps):
        raise ValueError("invalid --attempt-worker-caps")

    require_paths((args.td_root, args.tools_dir, args.rtl))
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / "coordinator.events.log"
    status_path = result_root / "coordinator.status.json"
    token = result_root.name.replace("-", "_")
    metadata = {
        "design": args.design,
        "rtl": str(args.rtl.resolve()),
        "tools_dir": str(args.tools_dir.resolve()),
        "tools_sha256": {
            name: sha256_file(args.tools_dir / name)
            for name in (
                "common.py",
                "run_sta.py",
                "run_hpc_sharded_pipeline.py",
                "package_dataset.py",
                "validate_dataset.py",
            )
        },
        "timing_startpoint_policy": "current_case_golden_nuiat_ports_only",
        "case_count": args.case_count,
        "hosts": hosts,
        "frontend_host": args.frontend_host,
        "package_host": args.package_host,
        "path_num": args.path_num,
        "endpoint_num": args.endpoint_num,
        "td_threads": args.td_threads,
        "attempt_worker_caps": caps,
        "scratch_parent": args.scratch_parent,
    }
    metadata_path = result_root / "coordinator.metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("existing coordinator metadata differs")
    else:
        atomic_write_json(metadata_path, metadata)

    projects = result_root / "generated" / "projects"
    if not (result_root / "generated" / "frontend.complete").is_file():
        write_status(status_path, "generating_frontend", **metadata)
        log("starting one-time frontend generation on {}".format(args.frontend_host), log_path)
        returncode = run_remote(
            args.frontend_host,
            FRONTEND_SCRIPT,
            (
                args.td_root.resolve(),
                args.tools_dir.resolve(),
                args.rtl.resolve(),
                args.design,
                args.case_count,
                result_root,
                token,
                args.scratch_parent,
            ),
            result_root / "frontend.remote.log",
        )
        if returncode:
            write_status(status_path, "frontend_failed", returncode=returncode, **metadata)
            log("frontend failed with status {}".format(returncode), log_path)
            return returncode
        log("frontend generation and netlist deduplication complete", log_path)

    missing = missing_cases(projects, args.design, args.case_count)
    for attempt, cap in enumerate(caps, 1):
        if not missing:
            break
        assignments = distribute(missing, hosts)
        workers = {
            host: len(indices) if cap == 0 else min(cap, len(indices))
            for host, indices in assignments.items()
        }
        write_status(
            status_path,
            "running_sta",
            attempt=attempt,
            missing_cases=missing,
            assignments=assignments,
            workers=workers,
            **metadata,
        )
        log(
            "STA attempt {}: {} cases across {} hosts, {} workers total".format(
                attempt, len(missing), len(assignments), sum(workers.values())
            ),
            log_path,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            futures = {}
            for host, indices in assignments.items():
                case_spec = ",".join(str(index) for index in indices)
                output = result_root / "shards" / host / "attempt_{}".format(attempt) / "remote.log"
                future = executor.submit(
                    run_remote,
                    host,
                    SHARD_SCRIPT,
                    (
                        args.td_root.resolve(),
                        args.tools_dir.resolve(),
                        args.design,
                        result_root,
                        token,
                        host,
                        case_spec,
                        workers[host],
                        args.path_num,
                        args.endpoint_num,
                        args.td_threads,
                        attempt,
                        args.scratch_parent,
                    ),
                    output,
                )
                futures[future] = host
            for future in concurrent.futures.as_completed(futures):
                host = futures[future]
                returncode = future.result()
                log(
                    "STA attempt {} on {} exited with status {}".format(
                        attempt, host, returncode
                    ),
                    log_path,
                )
        missing = missing_cases(projects, args.design, args.case_count)
        log(
            "STA attempt {} complete: {} labels valid, {} missing".format(
                attempt, args.case_count - len(missing), len(missing)
            ),
            log_path,
        )

    if missing:
        write_status(status_path, "sta_failed", missing_cases=missing, **metadata)
        log("STA exhausted retries; {} cases remain".format(len(missing)), log_path)
        return 1

    write_status(status_path, "packaging", **metadata)
    log("all labels valid; packaging on {}".format(args.package_host), log_path)
    returncode = run_remote(
        args.package_host,
        PACKAGE_SCRIPT,
        (
            args.tools_dir.resolve(),
            args.design,
            args.case_count,
            result_root,
            token,
            args.scratch_parent,
        ),
        result_root / "package.remote.log",
    )
    if returncode:
        write_status(status_path, "package_failed", returncode=returncode, **metadata)
        log("packaging failed with status {}".format(returncode), log_path)
        return returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = [
            executor.submit(
                run_remote,
                host,
                CLEANUP_SCRIPT,
                (token, args.scratch_parent, host),
                result_root / "cleanup" / (host + ".log"),
            )
            for host in hosts
        ]
        cleanup_codes = [future.result() for future in futures]
    write_status(
        status_path,
        "complete",
        valid_cases=args.case_count,
        cleanup_returncodes=dict(zip(hosts, cleanup_codes)),
        **metadata,
    )
    log("dataset complete: {}".format(result_root / "dataset"), log_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr, flush=True)
        raise
