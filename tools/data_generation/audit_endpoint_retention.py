#!/usr/bin/env python3
"""Explain how raw timing endpoints are reduced to parser-visible POs."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Set


LABEL_MARKER = "// pin to pin level synthesised"


def read_endpoint_labels(path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    in_labels = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line == LABEL_MARKER:
            in_labels = True
            continue
        if not in_labels or not line or line.startswith("//"):
            continue
        fields = line.replace(",", "").split()
        if len(fields) not in (2, 3):
            continue
        endpoint, value = fields[0], int(fields[-1])
        labels[endpoint] = max(labels.get(endpoint, value), value)
    return labels


def read_declared_outputs(path: Path) -> Set[str]:
    outputs: Set[str] = set()
    for raw_line in path.open("r", encoding="utf-8", errors="ignore"):
        line = raw_line.replace("\\", "").strip().rstrip(";").strip()
        if not line.startswith("output "):
            continue
        fields = line.split()
        if len(fields) == 2:
            outputs.add(fields[1])
            continue
        if len(fields) != 3:
            continue
        match = re.fullmatch(r"\[(-?\d+):(-?\d+)\]", fields[1])
        if match is None:
            continue
        high, low = int(match.group(1)), int(match.group(2))
        for bit in range(min(low, high), max(low, high) + 1):
            outputs.add("{}[{}]".format(fields[2], bit))
    return outputs


def sample(values: Set[str]) -> list[str]:
    return sorted(values)[:5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    with (args.dataset / "data.pkl").open("rb") as handle:
        dataset = pickle.load(handle)

    records = []
    for graph, info in sorted(dataset, key=lambda item: item[1]["design_name"]):
        design = info["design_name"]
        design_dir = args.raw_data / design
        labels = read_endpoint_labels(design_dir / (design + "_0") / "golden.txt")
        raw_endpoints = set(labels)
        declared_outputs = read_declared_outputs(design_dir / (design + ".v"))
        declared = raw_endpoints & declared_outputs
        old_reg_rule_removed = {
            name for name in declared if "reg" in name and not name.endswith("d")
        }
        vector_reg_d_false_rejections = {
            name
            for name in old_reg_rule_removed
            if re.search(r"_reg_d(?:\[\d+\])+$", name) is not None
        }
        reg_rule_removed = {
            name
            for name in declared
            if "reg" in re.sub(r"(?:\[\d+\])+$", "", name)
            and not re.sub(r"(?:\[\d+\])+$", "", name).endswith("d")
        }
        eligible = declared - reg_rule_removed

        nodes = set(info["nodes_name"])
        node_type = dict(zip(info["nodes_name"], info["ntype"]))
        level = {
            name: int(graph.ndata["level"][index].item())
            for index, name in enumerate(info["nodes_name"])
        }
        in_graph = eligible & nodes
        source_like = {
            name for name in in_graph if node_type[name] in ("input", "1'b0", "1'b1")
        }
        pre_abnormal = in_graph - source_like
        abnormal = {
            name for name in pre_abnormal if labels[name] == 0 and level[name] >= 2
        }
        expected_final = pre_abnormal - abnormal
        final = set(info["POs_name"])

        categories = {
            "raw_not_declared_output": raw_endpoints - declared_outputs,
            "reg_name_rule_removed": reg_rule_removed,
            "old_reg_name_rule_removed": old_reg_rule_removed,
            "vector_reg_d_false_rejections": vector_reg_d_false_rejections,
            "declared_eligible_not_in_filtered_graph": eligible - nodes,
            "source_like_output_removed": source_like,
            "zero_label_level_removed": abnormal,
            "expected_but_not_final": expected_final - final,
            "final_po": final,
        }
        records.append(
            {
                "design": design,
                "raw_endpoints": len(raw_endpoints),
                "declared_output_endpoints": len(declared),
                "eligible_after_reg_rule": len(eligible),
                "present_after_graph_filter": len(in_graph),
                "pre_abnormal_candidates": len(pre_abnormal),
                "final_po": len(final),
                "categories": {
                    key: {"count": len(values), "examples": sample(values)}
                    for key, values in categories.items()
                },
            }
        )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
