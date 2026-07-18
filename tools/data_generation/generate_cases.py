#!/usr/bin/env python3
"""Prepare and optionally run TangDynasty NUIAT case generation."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

from common import atomic_write_json, atomic_write_text, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-root", type=Path, required=True)
    parser.add_argument("--rtl", type=Path, required=True)
    parser.add_argument("--design-name")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sdcs", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-black-boxes", action="store_true")
    return parser.parse_args()


def expected_case_files(case_dir: Path) -> list[Path]:
    case = case_dir.name
    return [
        case_dir / (case + ".v"),
        case_dir / (case + ".sdc"),
        case_dir / (case + "_golden.txt"),
        case_dir / "run.tcl",
    ]


def collect_td_diagnostics(
    work_dir: Path, allow_black_boxes: bool = False
) -> list[str]:
    severity_pattern = re.compile(r"\b(?:ERROR|FATAL)\b", re.IGNORECASE)
    warning_pattern = re.compile(r"\bWARNING\b", re.IGNORECASE)
    black_box_pattern = re.compile(r"black box", re.IGNORECASE)
    diagnostics = []
    for log_file in sorted(work_dir.glob("td_*.log")):
        for line_number, line in enumerate(
            log_file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            is_black_box = black_box_pattern.search(line) is not None
            is_error = (
                severity_pattern.search(line) is not None
                and warning_pattern.search(line) is None
            )
            if is_black_box and allow_black_boxes:
                continue
            if is_black_box or is_error:
                diagnostics.append("{}:{}:{}".format(log_file.name, line_number, line))
                if len(diagnostics) >= 100:
                    return diagnostics
    return diagnostics


def main() -> int:
    args = parse_args()
    rtl = args.rtl.resolve()
    td_root = args.td_root.resolve()
    work_dir = args.work_dir.resolve()
    design = args.design_name or rtl.stem
    if not rtl.is_file():
        raise FileNotFoundError(rtl)
    if not design.replace("_", "").isalnum():
        raise ValueError("design name must contain only letters, digits, and underscores")
    if args.sdcs < 1:
        raise ValueError("--sdcs must be positive")

    input_dir = work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    staged_rtl = input_dir / (design + ".v")
    if staged_rtl.exists() and sha256_file(staged_rtl) != sha256_file(rtl):
        raise FileExistsError("staged RTL differs: {}".format(staged_rtl))
    if not staged_rtl.exists():
        shutil.copy2(str(rtl), str(staged_rtl))

    run_tcl = work_dir / "run.tcl"
    tcl_text = "generate_hkch_cases -dir {{{}}} -sdcs {}\nexit\n".format(
        input_dir, args.sdcs
    )
    if run_tcl.exists() and run_tcl.read_text(encoding="utf-8") != tcl_text:
        raise FileExistsError("existing run.tcl has different settings: {}".format(run_tcl))
    atomic_write_text(run_tcl, tcl_text)

    status = {
        "design": design,
        "rtl": str(rtl),
        "rtl_sha256": sha256_file(rtl),
        "sdcs": args.sdcs,
        "work_dir": str(work_dir),
        "allow_black_boxes": args.allow_black_boxes,
        "executed": False,
    }
    if not args.execute:
        atomic_write_json(work_dir / "generation_status.json", status)
        print("Prepared {}".format(run_tcl))
        return 0

    td_executable = td_root / "bin" / "td.sh"
    if not td_executable.is_file():
        raise FileNotFoundError(td_executable)
    existing_projects = work_dir / "projects"
    if existing_projects.exists() and any(existing_projects.iterdir()):
        complete = all(
            all(path.is_file() for path in expected_case_files(existing_projects / "{}_{}".format(design, index)))
            for index in range(args.sdcs)
        )
        if complete:
            status.update({"executed": True, "resumed": True, "returncode": 0})
            atomic_write_json(work_dir / "generation_status.json", status)
            print("All expected cases already exist; generation skipped")
            return 0
        raise FileExistsError("non-empty incomplete projects directory: {}".format(existing_projects))

    started = time.time()
    with (work_dir / "generation.stdout.log").open("w", encoding="utf-8") as stdout, (
        work_dir / "generation.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            [str(td_executable), str(run_tcl)],
            cwd=str(work_dir),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    missing = []
    for index in range(args.sdcs):
        case_dir = work_dir / "projects" / "{}_{}".format(design, index)
        missing.extend(str(path) for path in expected_case_files(case_dir) if not path.is_file())
    diagnostics = collect_td_diagnostics(work_dir)
    ignored_black_boxes = []
    if args.allow_black_boxes:
        ignored_black_boxes = [
            item for item in diagnostics if "black box" in item.lower()
        ]
        diagnostics = [
            item for item in diagnostics if "black box" not in item.lower()
        ]
    status.update(
        {
            "executed": True,
            "returncode": result.returncode,
            "wall_seconds": time.time() - started,
            "missing_outputs": missing,
            "td_diagnostics": diagnostics,
            "ignored_black_boxes": ignored_black_boxes,
        }
    )
    atomic_write_json(work_dir / "generation_status.json", status)
    if result.returncode != 0 or missing or diagnostics:
        print("Generation failed; inspect generation_status.json and TD logs")
        return 1
    print("Generated {} cases for {}".format(args.sdcs, design))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
