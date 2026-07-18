#!/usr/bin/env python3
"""Summarize complete labeled golden variants in generated case directories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import sha256_file, split_case_name, validate_labeled_golden


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-dir", type=Path, required=True)
    parser.add_argument("--design", action="append", required=True)
    parser.add_argument("--suffix", default="golden_new8.txt")
    args = parser.parse_args()

    result = {}
    for design in args.design:
        records = []
        endpoint_sets = []
        errors = []
        endpoint_intersection = None
        endpoint_union = set()
        netlist_hashes = set()
        pattern = "{}_*/{}_*_{}".format(design, design, args.suffix)
        for label in sorted(args.projects_dir.glob(pattern)):
            case_dir = label.parent
            _, case_index = split_case_name(case_dir.name)
            netlist = case_dir / (case_dir.name + ".v")
            try:
                validation = validate_labeled_golden(label)
            except Exception as error:
                errors.append({"case": case_index, "error": str(error)})
                continue
            endpoints = validation.pop("endpoints")
            endpoint_union.update(endpoints)
            if endpoint_intersection is None:
                endpoint_intersection = set(endpoints)
            else:
                endpoint_intersection.intersection_update(endpoints)
            if netlist.is_file():
                netlist_hashes.add(sha256_file(netlist))
            records.append({"case": case_index, **validation})
            endpoint_sets.append((case_index, validation["pi_count"], endpoints))

        pi_count_mode = None
        complete_case_intersection = None
        complete_cases = []
        if endpoint_sets:
            pi_count_mode = Counter(item[1] for item in endpoint_sets).most_common(1)[0][0]
            for case_index, pi_count, endpoints in endpoint_sets:
                if pi_count != pi_count_mode:
                    continue
                complete_cases.append(case_index)
                if complete_case_intersection is None:
                    complete_case_intersection = set(endpoints)
                else:
                    complete_case_intersection.intersection_update(endpoints)

        result[design] = {
            "valid_cases": records,
            "valid_case_count": len(records),
            "errors": errors,
            "endpoint_intersection": len(endpoint_intersection or set()),
            "endpoint_union": len(endpoint_union),
            "netlist_hash_count": len(netlist_hashes),
            "complete_pi_count": pi_count_mode,
            "complete_cases": sorted(complete_cases),
            "complete_case_count": len(complete_cases),
            "complete_endpoint_intersection": len(
                complete_case_intersection or set()
            ),
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
