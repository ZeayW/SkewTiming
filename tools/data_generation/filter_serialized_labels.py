#!/usr/bin/env python3
"""Create an auditable provisional dataset by masking suspect serialized labels."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path


REGISTER_PIN_RE = re.compile(
    r"^(?P<register>.+)_(?P<port>[qd])(?P<bits>(?:\[\d+\])*)$"
)


def is_same_register_q_to_d(begin: str, end: str) -> bool:
    begin_match = REGISTER_PIN_RE.fullmatch(begin)
    end_match = REGISTER_PIN_RE.fullmatch(end)
    if begin_match is None or end_match is None:
        return False
    return (
        "reg" in begin_match.group("register")
        and begin_match.group("port") == "q"
        and end_match.group("port") == "d"
        and begin_match.group("register") == end_match.group("register")
        and begin_match.group("bits") == end_match.group("bits")
    )


def filter_dataset(dataset, min_coverage: int):
    summary = {
        "status": "provisional_nonpublication",
        "min_endpoint_case_coverage": min_coverage,
        "designs": [],
    }
    for graph, info in dataset:
        cases = info["delay-label_pairs"]
        po_nids = graph.nodes()[graph.ndata["is_po"] == 1].tolist()
        po_index_by_nid = {nid: index for index, nid in enumerate(po_nids)}
        node_names = info["nodes_name"]
        base_labels = info["base_po_labels"]
        coverage = [0] * len(po_nids)
        for _, labels, _, _ in cases:
            for po_index, label in enumerate(labels):
                coverage[po_index] += label >= 0

        raw_rule_counts = defaultdict(int)
        exclusive_counts = defaultdict(int)
        labeled_before = sum(coverage)
        labeled_after = 0

        for case_position, case in enumerate(cases):
            pi_delays, labels, residuals, edges = case
            sources, destinations, weights = edges
            edges_by_po = defaultdict(list)
            for edge_index, destination in enumerate(destinations):
                po_index = po_index_by_nid[destination]
                edges_by_po[po_index].append(edge_index)

            for po_index, label in enumerate(labels):
                if label < 0:
                    continue
                reasons = []
                if coverage[po_index] < min_coverage:
                    reasons.append("low_coverage")
                base_label = base_labels[po_index]
                if base_label >= 0 and label < base_label:
                    reasons.append("below_zero_input_baseline")
                endpoint_edges = edges_by_po.get(po_index, [])
                if endpoint_edges and all(
                    is_same_register_q_to_d(
                        node_names[sources[edge_index]],
                        node_names[destinations[edge_index]],
                    )
                    for edge_index in endpoint_edges
                ):
                    reasons.append("same_bit_register_self_path")

                for reason in reasons:
                    raw_rule_counts[reason] += 1
                if reasons:
                    exclusive_counts[reasons[0]] += 1
                    labels[po_index] = -1
                    residuals[po_index] = -1
                else:
                    labeled_after += 1
            cases[case_position] = (
                pi_delays,
                labels,
                residuals,
                edges,
            )

        summary["designs"].append(
            {
                "design": info["design_name"],
                "cases": len(cases),
                "endpoints": len(po_nids),
                "labeled_before": labeled_before,
                "labeled_after": labeled_after,
                "masked": labeled_before - labeled_after,
                "raw_rule_matches": dict(raw_rule_counts),
                "exclusive_mask_reason": dict(exclusive_counts),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="source dataset directory")
    parser.add_argument("output", type=Path, help="new dataset directory")
    parser.add_argument("--min-coverage", type=int, default=20)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    with (source / "data.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    summary = filter_dataset(dataset, args.min_coverage)
    summary["source"] = str(source)

    with (output / "data.pkl").open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for filename in ("split.pkl",):
        source_file = source / filename
        if source_file.is_file():
            shutil.copy2(source_file, output / filename)
    with (output / "filter_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
