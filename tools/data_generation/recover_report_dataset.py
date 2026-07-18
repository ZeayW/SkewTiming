#!/usr/bin/env python3
"""Recover an explicitly provisional dataset from retained timing reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from common import (
    atomic_write_json,
    extract_first_module,
    iter_timing_paths,
    normalize_input_levels,
    parse_input_ranges,
    read_input_levels,
    sha256_file,
    split_case_name,
    validate_labeled_golden,
    write_labeled_golden,
)


class LabelAccumulator:
    def __init__(self, input_levels: Mapping[str, int]) -> None:
        self.input_levels = input_levels
        self.best: Dict[str, int] = {}
        self.critical: Dict[str, Set[Tuple[str, int]]] = {}
        self.path_count = 0
        self.matched_path_count = 0
        self.unmatched_path_count = 0

    def update(self, begin: str, end: str, logic_level: int) -> None:
        self.path_count += 1
        if begin not in self.input_levels:
            self.unmatched_path_count += 1
            return
        self.matched_path_count += 1
        total = self.input_levels[begin] + logic_level
        if end not in self.best or total > self.best[end]:
            self.best[end] = total
            self.critical[end] = {(begin, total)}
        elif total == self.best[end]:
            self.critical[end].add((begin, total))

    def labels(self, endpoints: Optional[Set[str]] = None) -> List[Tuple[str, str, int]]:
        selected = set(self.critical) if endpoints is None else endpoints
        return [
            (endpoint, begin, total)
            for endpoint in sorted(selected)
            for begin, total in sorted(self.critical[endpoint])
        ]


def _case_file(case_dir: Path, suffix: str) -> Path:
    return case_dir / (case_dir.name + suffix)


def _read_partial_case_inputs(path: Path) -> Tuple[List[Tuple[str, int]], int]:
    """Read the valid prefix of an interrupted historical constraint file."""

    inputs: List[Tuple[str, int]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.replace("\x00", "").strip()
            if line == "// pin to pin level synthesised":
                break
            if not line or line.startswith("//"):
                continue
            fields = line.split()
            if len(fields) != 2:
                skipped += 1
                continue
            try:
                inputs.append((fields[0], int(fields[1])))
            except ValueError:
                skipped += 1
    if not inputs:
        raise ValueError("no recoverable PI labels found in {}".format(path))
    return inputs, skipped


def _merged_inputs(
    reference: Sequence[Tuple[str, int]], overrides: Iterable[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    override_map = dict(overrides)
    merged = [(pin, override_map.get(pin, 0)) for pin, _ in reference]
    known = {pin for pin, _ in reference}
    merged.extend((pin, level) for pin, level in override_map.items() if pin not in known)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--reference-golden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dirs = [path.resolve() for path in args.case_dir]
    output_root = args.output_dir.resolve()
    design_dir = output_root / args.design_name
    if design_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(design_dir))

    source_netlist = _case_file(case_dirs[0], ".v")
    input_ranges = parse_input_ranges(source_netlist)
    reference_inputs = normalize_input_levels(
        read_input_levels(args.reference_golden.resolve()), input_ranges
    )
    zero_inputs = [(pin, 0) for pin, _ in reference_inputs]
    zero_levels = dict(zero_inputs)
    baseline = LabelAccumulator(zero_levels)
    case_records = []
    case_accumulators: List[Tuple[Path, List[Tuple[str, int]], LabelAccumulator]] = []

    for case_dir in case_dirs:
        original_design, original_index = split_case_name(case_dir.name)
        source_golden = _case_file(case_dir, "_golden.txt")
        report = case_dir / "timing.rpt"
        if not source_golden.is_file() or not report.is_file():
            raise FileNotFoundError("missing golden/report in {}".format(case_dir))
        partial_inputs, skipped_input_lines = _read_partial_case_inputs(source_golden)
        case_inputs = _merged_inputs(
            reference_inputs,
            normalize_input_levels(partial_inputs, input_ranges),
        )
        accumulator = LabelAccumulator(dict(case_inputs))
        for begin, end, logic_level in iter_timing_paths(report):
            baseline.update(begin, end, logic_level)
            accumulator.update(begin, end, logic_level)
        case_accumulators.append((case_dir, case_inputs, accumulator))
        case_records.append(
            {
                "source_case": case_dir.name,
                "source_design": original_design,
                "source_index": original_index,
                "report": str(report),
                "report_bytes": report.stat().st_size,
                "paths": accumulator.path_count,
                "matched_paths": accumulator.matched_path_count,
                "unmatched_paths": accumulator.unmatched_path_count,
                "reported_endpoints": len(accumulator.best),
                "skipped_input_lines": skipped_input_lines,
            }
        )

    common_endpoints = set(baseline.best)
    for _, _, accumulator in case_accumulators:
        common_endpoints.intersection_update(accumulator.best)
    if not common_endpoints:
        raise ValueError("reports have no common labeled endpoints")

    target_netlist = design_dir / (args.design_name + ".v")
    extract_first_module(source_netlist, target_netlist)
    base_label = design_dir / (args.design_name + "_0") / "golden.txt"
    write_labeled_golden(base_label, zero_inputs, baseline.labels(common_endpoints))
    base_validation = validate_labeled_golden(base_label)

    for output_index, (_, case_inputs, accumulator) in enumerate(case_accumulators, start=1):
        target = design_dir / (args.design_name + "_{}".format(output_index)) / "golden.txt"
        write_labeled_golden(target, case_inputs, accumulator.labels(common_endpoints))
        validation = validate_labeled_golden(target)
        case_records[output_index - 1].update(
            {
                "output_case": output_index,
                "pi_count": validation["pi_count"],
                "endpoint_count": validation["endpoint_count"],
                "label_count": validation["label_count"],
            }
        )

    manifest = {
        "status": "provisional_nonpublication",
        "warning": (
            "The zero-delay baseline is reconstructed only from paths retained in the "
            "listed nonzero-NUIAT reports; it is not a fresh zero-delay STA result."
        ),
        "design": args.design_name,
        "netlist": str(source_netlist),
        "netlist_sha256": sha256_file(target_netlist),
        "reference_golden": str(args.reference_golden.resolve()),
        "reference_pi_count": len(reference_inputs),
        "common_endpoint_count": len(common_endpoints),
        "baseline": {
            "pi_count": base_validation["pi_count"],
            "endpoint_count": base_validation["endpoint_count"],
            "label_count": base_validation["label_count"],
            "matched_paths": baseline.matched_path_count,
        },
        "cases": case_records,
    }
    atomic_write_json(design_dir / "recovery_manifest.json", manifest)
    print(
        "recovered {}: {} cases, {} PIs, {} common endpoints".format(
            args.design_name, len(case_records), len(reference_inputs), len(common_endpoints)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
