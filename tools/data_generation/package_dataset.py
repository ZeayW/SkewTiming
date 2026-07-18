#!/usr/bin/env python3
"""Package generated netlists and labels into the round7 parser layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

from common import (
    atomic_write_json,
    extract_first_module,
    iter_case_dirs,
    parse_index_spec,
    sha256_file,
    split_case_name,
    validate_labeled_golden,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", action="append")
    parser.add_argument("--cases", help="comma-separated indices/ranges")
    parser.add_argument("--label-name", default="golden_labeled.txt")
    parser.add_argument("--keep-full-netlist", action="store_true")
    parser.add_argument(
        "--endpoint-mode",
        choices=("strict", "union"),
        default="strict",
        help=(
            "strict requires identical endpoint sets; union preserves each "
            "case's labels and lets the parser represent missing labels as -1"
        ),
    )
    return parser.parse_args()


def copy_if_same_or_missing(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError("destination differs: {}".format(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def main() -> int:
    args = parse_args()
    projects_dir = args.projects_dir.resolve()
    output_dir = args.output_dir.resolve()
    designs: Optional[Set[str]] = set(args.design) if args.design else None
    indices = parse_index_spec(args.cases)
    case_dirs = iter_case_dirs(projects_dir, designs, indices)
    if not case_dirs:
        raise ValueError("no matching cases")

    grouped: Dict[str, List[Path]] = {}
    for case_dir in case_dirs:
        design, _ = split_case_name(case_dir.name)
        grouped.setdefault(design, []).append(case_dir)

    manifest: Dict[str, object] = {"projects_dir": str(projects_dir), "designs": {}}
    for design, design_cases in sorted(grouped.items()):
        design_cases.sort(key=lambda path: split_case_name(path.name)[1])
        source_netlist = design_cases[0] / (design_cases[0].name + ".v")
        target_netlist = output_dir / design / (design + ".v")
        if target_netlist.exists():
            pass
        elif args.keep_full_netlist:
            copy_if_same_or_missing(source_netlist, target_netlist)
        else:
            extract_first_module(source_netlist, target_netlist)

        case_records = []
        endpoint_reference = None
        endpoint_union: Set[str] = set()
        endpoint_intersection: Optional[Set[str]] = None
        endpoint_variants = set()
        for case_dir in design_cases:
            label = case_dir / args.label_name
            if not label.is_file():
                raise FileNotFoundError(label)
            validation = validate_labeled_golden(
                label, require_nuiat_startpoints=True
            )
            endpoints = set(validation.pop("endpoints"))
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
                raise ValueError("endpoint set differs in {}".format(case_dir))
            target = output_dir / design / case_dir.name / "golden.txt"
            copy_if_same_or_missing(label, target)
            case_records.append(
                {
                    "case": case_dir.name,
                    "label_sha256": sha256_file(label),
                    **validation,
                }
            )
        manifest["designs"][design] = {
            "case_count": len(case_records),
            "endpoint_mode": args.endpoint_mode,
            "selected_endpoint_count": len(
                endpoint_reference if args.endpoint_mode == "strict" else endpoint_union
            ),
            "endpoint_union_count": len(endpoint_union),
            "common_endpoint_count": len(endpoint_intersection or set()),
            "endpoint_variant_count": len(endpoint_variants),
            "netlist_sha256": sha256_file(target_netlist),
            "cases": case_records,
        }
        print(
            "packaged {}: {} cases, {} endpoint mode, {} common, {} union".format(
                design,
                len(case_records),
                args.endpoint_mode,
                len(endpoint_intersection or set()),
                len(endpoint_union),
            )
        )

    atomic_write_json(output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
