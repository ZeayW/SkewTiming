#!/usr/bin/env python3
"""Validate a packaged round7-style NUA-Timer raw dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

from common import atomic_write_json, split_case_name, validate_labeled_golden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument(
        "--endpoint-mode", choices=("strict", "union"), default="strict"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    summaries: Dict[str, object] = {}
    errors: List[str] = []
    for design_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        design = design_dir.name
        netlist = design_dir / (design + ".v")
        if not netlist.is_file():
            errors.append("{}: missing netlist".format(design))
            continue
        cases = []
        for case_dir in design_dir.iterdir():
            if not case_dir.is_dir():
                continue
            try:
                case_design, index = split_case_name(case_dir.name)
            except ValueError:
                continue
            if case_design != design:
                errors.append("{}: mismatched case {}".format(design, case_dir.name))
                continue
            cases.append((index, case_dir))
        cases.sort()
        if args.expected_cases is not None and len(cases) != args.expected_cases:
            errors.append(
                "{}: expected {} cases, found {}".format(
                    design, args.expected_cases, len(cases)
                )
            )
        endpoint_reference: Optional[Set[str]] = None
        endpoint_union: Set[str] = set()
        endpoint_intersection: Optional[Set[str]] = None
        endpoint_variants = set()
        pi_counts = []
        for _, case_dir in cases:
            label = case_dir / "golden.txt"
            try:
                validation = validate_labeled_golden(
                    label, require_nuiat_startpoints=True
                )
            except Exception as error:
                errors.append("{}: {}".format(case_dir.name, error))
                continue
            endpoints = set(validation["endpoints"])
            endpoint_union.update(endpoints)
            endpoint_intersection = (
                set(endpoints)
                if endpoint_intersection is None
                else endpoint_intersection.intersection(endpoints)
            )
            endpoint_variants.add(tuple(sorted(endpoints)))
            if endpoint_reference is None:
                endpoint_reference = endpoints
            elif args.endpoint_mode == "strict" and endpoints != endpoint_reference:
                errors.append("{}: endpoint set differs".format(case_dir.name))
            pi_counts.append(validation["pi_count"])
        summaries[design] = {
            "case_count": len(cases),
            "endpoint_mode": args.endpoint_mode,
            "endpoint_count": len(
                endpoint_reference
                if args.endpoint_mode == "strict"
                else endpoint_union
            ),
            "endpoint_union_count": len(endpoint_union),
            "common_endpoint_count": len(endpoint_intersection or set()),
            "endpoint_variant_count": len(endpoint_variants),
            "pi_count_min": min(pi_counts) if pi_counts else 0,
            "pi_count_max": max(pi_counts) if pi_counts else 0,
            "netlist_bytes": netlist.stat().st_size,
        }
    result = {"dataset_dir": str(dataset_dir), "designs": summaries, "errors": errors}
    atomic_write_json(dataset_dir / "validation_summary.json", result)
    for design, summary in summaries.items():
        print(
            "{design}: cases={case_count}, PIs={pi_count_min}-{pi_count_max}, "
            "endpoints={endpoint_count}, mode={endpoint_mode}, "
            "common={common_endpoint_count}, union={endpoint_union_count}".format(
                design=design, **summary
            )
        )
    if errors:
        print("Validation failed with {} errors".format(len(errors)))
        return 1
    print("Dataset validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
