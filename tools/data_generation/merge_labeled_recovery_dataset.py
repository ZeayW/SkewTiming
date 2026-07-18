#!/usr/bin/env python3
"""Merge complete labeled variants with report-recovered cases."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common import (
    LABEL_MARKER,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    split_case_name,
    validate_labeled_golden,
)


def filter_golden(source: Path, destination: Path, endpoints: Set[str]) -> None:
    text = source.read_text(encoding="utf-8", errors="ignore")
    before, after = text.split(LABEL_MARKER, 1)
    labels = []
    for raw_line in after.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        endpoint = line.replace(",", "").split()[0]
        if endpoint in endpoints:
            labels.append(line)
    output = before.rstrip() + "\n" + LABEL_MARKER + "\n"
    output += "// outpin, critical input, max level\n"
    output += "\n".join(labels) + "\n"
    atomic_write_text(destination, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-dir", type=Path, required=True)
    parser.add_argument("--recovered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", action="append", required=True)
    parser.add_argument("--suffix", default="golden_new8.txt")
    parser.add_argument(
        "--endpoint-mode",
        choices=("intersection", "union"),
        default="intersection",
        help=(
            "intersection writes the same common endpoint set to every case; "
            "union preserves each case's labels for parser-side -1 padding"
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))

    output_manifest: Dict[str, object] = {
        "status": "provisional_nonpublication",
        "warning": (
            "The zero-input baseline is reconstructed from retained reports; "
            "it is not a fresh zero-delay STA result."
        ),
        "designs": {},
    }

    for design in args.design:
        recovered_design = args.recovered_dir / design
        recovery_manifest = json.loads(
            (recovered_design / "recovery_manifest.json").read_text()
        )
        expected_pi_count = int(recovery_manifest["reference_pi_count"])
        sources: Dict[int, Tuple[Path, str]] = {
            0: (recovered_design / (design + "_0") / "golden.txt", "baseline")
        }
        for record in recovery_manifest["cases"]:
            output_case = int(record["output_case"])
            source_case = int(record["source_index"])
            sources[source_case] = (
                recovered_design / (design + "_{}".format(output_case)) / "golden.txt",
                "recovered_report",
            )

        pattern = "{}_*/{}_*_{}".format(design, design, args.suffix)
        skipped = []
        for label in sorted(args.projects_dir.glob(pattern)):
            _, case_index = split_case_name(label.parent.name)
            try:
                validation = validate_labeled_golden(label)
            except Exception as error:
                skipped.append({"case": case_index, "reason": str(error)})
                continue
            if validation["pi_count"] != expected_pi_count:
                skipped.append(
                    {
                        "case": case_index,
                        "reason": "PI count {} != {}".format(
                            validation["pi_count"], expected_pi_count
                        ),
                    }
                )
                continue
            sources[case_index] = (label, "labeled_golden")

        endpoint_intersection = None
        endpoint_union: Set[str] = set()
        source_endpoints: Dict[int, Set[str]] = {}
        source_records: List[Dict[str, object]] = []
        for case_index, (label, source_type) in sorted(sources.items()):
            validation = validate_labeled_golden(label)
            endpoints = validation.pop("endpoints")
            if validation["pi_count"] != expected_pi_count:
                raise ValueError("unexpected PI count in {}".format(label))
            if endpoint_intersection is None:
                endpoint_intersection = set(endpoints)
            else:
                endpoint_intersection.intersection_update(endpoints)
            endpoint_union.update(endpoints)
            source_endpoints[case_index] = set(endpoints)
            source_records.append(
                {
                    "case": case_index,
                    "source": str(label),
                    "source_type": source_type,
                    **validation,
                }
            )

        common_endpoints = endpoint_intersection or set()
        if not common_endpoints:
            raise ValueError("{} has no endpoints common to all cases".format(design))
        selected_endpoints = (
            common_endpoints if args.endpoint_mode == "intersection" else endpoint_union
        )

        target_design = args.output_dir / design
        target_design.mkdir(parents=True)
        shutil.copy2(recovered_design / (design + ".v"), target_design / (design + ".v"))
        for case_index, (label, _) in sorted(sources.items()):
            target = target_design / (design + "_{}".format(case_index)) / "golden.txt"
            retained_endpoints = (
                common_endpoints
                if args.endpoint_mode == "intersection"
                else source_endpoints[case_index]
            )
            filter_golden(label, target, retained_endpoints)
            validation = validate_labeled_golden(target)
            actual = set(validation["endpoints"])
            if actual != retained_endpoints:
                raise ValueError(
                    "filtered endpoint mismatch in {}: expected {}, actual {}, "
                    "missing {}, extra {}".format(
                        target,
                        len(retained_endpoints),
                        len(actual),
                        sorted(retained_endpoints - actual)[:5],
                        sorted(actual - retained_endpoints)[:5],
                    )
                )

        output_manifest["designs"][design] = {
            "case_count": len(sources),
            "cases": source_records,
            "endpoint_mode": args.endpoint_mode,
            "selected_endpoint_count": len(selected_endpoints),
            "endpoint_union_count": len(endpoint_union),
            "common_endpoint_count": len(common_endpoints),
            "pi_count": expected_pi_count,
            "netlist_sha256": sha256_file(target_design / (design + ".v")),
            "skipped_variants": skipped,
        }
        print(
            "merged {}: {} cases, {} PIs, {} {} endpoints "
            "({} common, {} union)".format(
                design,
                len(sources),
                expected_pi_count,
                len(selected_endpoints),
                args.endpoint_mode,
                len(common_endpoints),
                len(endpoint_union),
            )
        )

    atomic_write_json(args.output_dir / "merge_manifest.json", output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
