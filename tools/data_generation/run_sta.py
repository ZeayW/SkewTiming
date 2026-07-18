#!/usr/bin/env python3
"""Run TangDynasty STA for generated cases and create complete labels."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

from common import (
    atomic_write_json,
    atomic_write_text,
    build_read_checkpoint_tcl,
    build_endpoint_labels,
    iter_timing_paths,
    iter_case_dirs,
    normalize_input_levels,
    parse_index_spec,
    parse_input_ranges,
    prepare_sta_netlist,
    read_input_levels,
    rewrite_sta_tcl,
    sha256_td_netlist_content,
    split_case_name,
    validate_labeled_golden,
    write_labeled_golden,
)


DEFAULT_PATH_NUM = 1
DEFAULT_ENDPOINT_NUM = 200000
DEFAULT_TD_THREADS = "auto"


@dataclass(frozen=True)
class CaseFlow:
    netlist: Optional[Path]
    checkpoint: Optional[Path]
    content_sha256: str
    preparation: Mapping[str, int]
    sta_netlist_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-root", type=Path, required=True)
    parser.add_argument("--projects-dir", type=Path, required=True)
    parser.add_argument("--design", action="append")
    parser.add_argument("--cases", help="comma-separated indices/ranges, e.g. 0-9,20")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--path-num", type=int, default=DEFAULT_PATH_NUM)
    parser.add_argument("--endpoint-num", type=int, default=DEFAULT_ENDPOINT_NUM)
    parser.add_argument(
        "--td-threads",
        choices=("auto", "1", "2", "4", "8", "16", "32"),
        default=DEFAULT_TD_THREADS,
    )
    parser.add_argument("--no-read-checkpoint", action="store_true")
    parser.add_argument("--qor-monitor", action="store_true")
    parser.add_argument("--output-name", default="golden_labeled.txt")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--timeout", type=int, default=0, help="seconds per case; 0 disables")
    return parser.parse_args()


def build_range_cache(case_dirs: List[Path]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    cache: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for case_dir in case_dirs:
        design, _ = split_case_name(case_dir.name)
        if design in cache:
            continue
        netlist = case_dir / (case_dir.name + ".v")
        cache[design] = parse_input_ranges(netlist)
    return cache


def _checkpoint_is_reusable(
    status_path: Path,
    checkpoint: Path,
    content_sha256: str,
    td_threads: str,
) -> bool:
    if (
        not status_path.is_file()
        or not checkpoint.is_file()
        or checkpoint.stat().st_size == 0
    ):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        status.get("state") == "complete"
        and status.get("content_sha256") == content_sha256
        and status.get("td_threads") == td_threads
    )


def _build_read_checkpoint(
    design: str,
    case_dir: Path,
    canonical_netlist: Path,
    content_sha256: str,
    td_executable: Path,
    cache_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Dict[str, object]]:
    checkpoint = cache_dir / "read.db"
    status_path = cache_dir / "read_checkpoint.status.json"
    if _checkpoint_is_reusable(
        status_path, checkpoint, content_sha256, args.td_threads
    ):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["reused"] = True
        return checkpoint, status

    checkpoint.unlink(missing_ok=True)
    tcl_path = cache_dir / "build_read_checkpoint.tcl"
    atomic_write_text(
        tcl_path,
        build_read_checkpoint_tcl(
            (case_dir / "run.tcl").read_text(encoding="utf-8"),
            canonical_netlist,
            checkpoint,
            td_threads=args.td_threads,
        ),
    )
    started = time.time()
    with (cache_dir / "read_checkpoint.stdout.log").open(
        "w", encoding="utf-8"
    ) as stdout, (cache_dir / "read_checkpoint.stderr.log").open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            [str(td_executable), tcl_path.name],
            cwd=str(cache_dir),
            stdout=stdout,
            stderr=stderr,
            timeout=args.timeout or None,
            check=False,
        )
    status = {
        "design": design,
        "state": "complete" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_seconds": time.time() - started,
        "content_sha256": content_sha256,
        "td_threads": args.td_threads,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "reused": False,
    }
    if result.returncode != 0 or status["checkpoint_bytes"] == 0:
        status["state"] = "failed"
        atomic_write_json(status_path, status)
        raise RuntimeError(
            "failed to build read checkpoint for {}; inspect {}".format(
                design, cache_dir
            )
        )
    atomic_write_json(status_path, status)
    return checkpoint, status


def build_case_flows(
    case_dirs: List[Path],
    projects_dir: Path,
    td_executable: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[Path, CaseFlow], List[Dict[str, object]]]:
    grouped: Dict[str, List[Path]] = {}
    for case_dir in case_dirs:
        design, _ = split_case_name(case_dir.name)
        grouped.setdefault(design, []).append(case_dir)

    cache_root = projects_dir / ".nua_read_cache"
    flows: Dict[Path, CaseFlow] = {}
    cache_statuses: List[Dict[str, object]] = []
    for design, design_cases in sorted(grouped.items()):
        design_cases.sort(key=lambda path: split_case_name(path.name)[1])
        hashes = {
            sha256_td_netlist_content(case_dir / (case_dir.name + ".v"))
            for case_dir in design_cases
        }
        if len(hashes) != 1:
            raise ValueError(
                "{} has {} distinct netlists after ignoring TD timestamps".format(
                    design, len(hashes)
                )
            )
        content_sha256 = next(iter(hashes))
        cache_dir = cache_root / design
        cache_dir.mkdir(parents=True, exist_ok=True)
        canonical_netlist = cache_dir / (design + ".v")
        source_netlist = design_cases[0] / (design_cases[0].name + ".v")
        preparation = prepare_sta_netlist(source_netlist, canonical_netlist)

        checkpoint = None
        checkpoint_status: Dict[str, object] = {
            "design": design,
            "state": "not_needed",
            "content_sha256": content_sha256,
            "case_count": len(design_cases),
        }
        if len(design_cases) > 1 and not args.no_read_checkpoint:
            checkpoint, checkpoint_status = _build_read_checkpoint(
                design,
                design_cases[0],
                canonical_netlist,
                content_sha256,
                td_executable,
                cache_dir,
                args,
            )
            checkpoint_status["case_count"] = len(design_cases)
        cache_statuses.append(checkpoint_status)

        flow = CaseFlow(
            netlist=None if checkpoint is not None else canonical_netlist,
            checkpoint=checkpoint,
            content_sha256=content_sha256,
            preparation=preparation,
            sta_netlist_bytes=canonical_netlist.stat().st_size,
        )
        for case_dir in design_cases:
            flows[case_dir] = flow
    return flows, cache_statuses


def process_case(
    case_dir: Path,
    td_executable: Path,
    input_ranges: Dict[str, Tuple[int, int]],
    flow: Optional[CaseFlow],
    args: argparse.Namespace,
) -> Dict[str, object]:
    case = case_dir.name
    output = case_dir / args.output_name
    status_path = case_dir / "nua_label.status.json"
    started = time.time()
    netlist_status: Dict[str, object] = {}
    try:
        if output.is_file():
            validation = validate_labeled_golden(
                output, require_nuiat_startpoints=True
            )
            return {"case": case, "state": "skipped", **validation}

        golden = case_dir / (case + "_golden.txt")
        run_tcl = case_dir / "run.tcl"
        netlist = case_dir / (case + ".v")
        for required in (golden, run_tcl, netlist, case_dir / (case + ".sdc")):
            if not required.is_file():
                raise FileNotFoundError(required)

        if flow is None:
            raise RuntimeError("missing prepared flow for {}".format(case))
        full_netlist_bytes = netlist.stat().st_size
        netlist_status = {
            "full_netlist_bytes": full_netlist_bytes,
            "sta_netlist_bytes": flow.sta_netlist_bytes,
            "netlist_content_sha256": flow.content_sha256,
            "read_checkpoint": str(flow.checkpoint) if flow.checkpoint else None,
            **flow.preparation,
        }

        inputs = normalize_input_levels(read_input_levels(golden), input_ranges)
        temporary_tcl = case_dir / ".nua_label.tcl"
        atomic_write_text(
            temporary_tcl,
            rewrite_sta_tcl(
                run_tcl.read_text(encoding="utf-8"),
                args.path_num,
                args.endpoint_num,
                [pin for pin, _ in inputs],
                checkpoint=flow.checkpoint,
                netlist=flow.netlist,
                keep_debug_netlists=args.keep_intermediates,
                qor_monitor=args.qor_monitor,
                td_threads=args.td_threads,
            ),
        )
        td_started = time.time()
        with (case_dir / "nua_label.stdout.log").open("w", encoding="utf-8") as stdout, (
            case_dir / "nua_label.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                [str(td_executable), temporary_tcl.name],
                cwd=str(case_dir),
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout or None,
                check=False,
            )
        td_wall_seconds = time.time() - td_started
        if result.returncode != 0:
            raise RuntimeError("TD exited with status {}".format(result.returncode))
        report = case_dir / "timing.rpt"
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("TD did not create a non-empty timing.rpt")
        report_bytes = report.stat().st_size
        parse_started = time.time()
        label_diagnostics: Dict[str, int] = {}
        labels = build_endpoint_labels(
            inputs,
            iter_timing_paths(report),
            diagnostics=label_diagnostics,
        )
        report_parse_seconds = time.time() - parse_started
        write_labeled_golden(output, inputs, labels)
        validation = validate_labeled_golden(
            output, require_nuiat_startpoints=True
        )
        status = {
            "case": case,
            "state": "complete",
            "wall_seconds": time.time() - started,
            "td_wall_seconds": td_wall_seconds,
            "report_bytes": report_bytes,
            "report_parse_seconds": report_parse_seconds,
            "requested_nuiat_startpoint_count": len(inputs),
            **label_diagnostics,
            **netlist_status,
            **validation,
        }
        atomic_write_json(status_path, status)
        if not args.keep_intermediates:
            for generated in (temporary_tcl, report, case_dir / "read.v", case_dir / "gate.v"):
                generated.unlink(missing_ok=True)
        return status
    except Exception as error:  # Keep all intermediate files for diagnosis.
        status = {
            "case": case,
            "state": "failed",
            "wall_seconds": time.time() - started,
            "error": "{}: {}".format(type(error).__name__, error),
            **netlist_status,
        }
        atomic_write_json(status_path, status)
        return status


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.path_num < 1:
        raise ValueError("--path-num must be positive")
    if args.endpoint_num < 1:
        raise ValueError("--endpoint-num must be positive")
    projects_dir = args.projects_dir.resolve()
    td_executable = args.td_root.resolve() / "bin" / "td.sh"
    if not td_executable.is_file():
        raise FileNotFoundError(td_executable)
    designs: Optional[Set[str]] = set(args.design) if args.design else None
    indices = parse_index_spec(args.cases)
    case_dirs = iter_case_dirs(projects_dir, designs, indices)
    if not case_dirs:
        raise ValueError("no matching case directories in {}".format(projects_dir))
    range_cache = build_range_cache(case_dirs)
    pending_case_dirs = [
        case_dir
        for case_dir in case_dirs
        if not (case_dir / args.output_name).is_file()
    ]
    case_flows, cache_statuses = build_case_flows(
        pending_case_dirs, projects_dir, td_executable, args
    )

    results: List[Dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for case_dir in case_dirs:
            design, _ = split_case_name(case_dir.name)
            futures.append(
                executor.submit(
                    process_case,
                    case_dir,
                    td_executable,
                    range_cache[design],
                    case_flows.get(case_dir),
                    args,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print("{state:8s} {case}".format(**result), flush=True)

    results.sort(key=lambda item: split_case_name(str(item["case"])))
    summary = {
        "projects_dir": str(projects_dir),
        "workers": args.workers,
        "path_num": args.path_num,
        "endpoint_num": args.endpoint_num,
        "td_threads": args.td_threads,
        "read_checkpoint": not args.no_read_checkpoint,
        "qor_monitor": args.qor_monitor,
        "keep_intermediates": args.keep_intermediates,
        "requested_cases": len(case_dirs),
        "complete": sum(item["state"] == "complete" for item in results),
        "skipped": sum(item["state"] == "skipped" for item in results),
        "failed": sum(item["state"] == "failed" for item in results),
        "read_checkpoints": cache_statuses,
        "cases": results,
    }
    atomic_write_json(projects_dir / "nua_label.summary.json", summary)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
