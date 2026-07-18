#!/usr/bin/env python3
"""Inspect graph and label dimensions in a serialized NUA-Timer dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    with (args.dataset / "data.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    records = []
    for graph, info in dataset:
        pi_count = int(graph.ndata["is_pi"].sum().item())
        po_count = int(graph.ndata["is_po"].sum().item())
        po_nids = graph.nodes()[graph.ndata["is_po"] == 1].tolist()
        po_index_by_nid = {nid: index for index, nid in enumerate(po_nids)}
        cases = info["delay-label_pairs"]
        labeled_counts = []
        missing_counts = []
        for case_index, (pi_delay, po_label, po_residual, pi2po_edges) in enumerate(cases):
            if len(pi_delay) != pi_count:
                raise ValueError("{} case {} PI mismatch".format(info["design_name"], case_index))
            if len(po_label) != po_count or len(po_residual) != po_count:
                raise ValueError("{} case {} PO mismatch".format(info["design_name"], case_index))
            invalid_labels = [label for label in po_label if label < 0 and label != -1]
            if invalid_labels:
                raise ValueError(
                    "{} case {} has invalid negative labels {}".format(
                        info["design_name"], case_index, invalid_labels[:5]
                    )
                )
            missing_count = sum(label == -1 for label in po_label)
            labeled_count = po_count - missing_count
            if labeled_count == 0:
                raise ValueError(
                    "{} case {} has no labeled endpoints".format(
                        info["design_name"], case_index
                    )
                )
            labeled_counts.append(labeled_count)
            missing_counts.append(missing_count)

            sources, destinations, weights = pi2po_edges
            if not (len(sources) == len(destinations) == len(weights)):
                raise ValueError(
                    "{} case {} has misaligned pi2po arrays".format(
                        info["design_name"], case_index
                    )
                )
            for destination in destinations:
                po_index = po_index_by_nid.get(destination)
                if po_index is None:
                    raise ValueError(
                        "{} case {} has pi2po edge to non-PO {}".format(
                            info["design_name"], case_index, destination
                        )
                    )
                if po_label[po_index] == -1:
                    raise ValueError(
                        "{} case {} has pi2po edge to an unlabeled PO".format(
                            info["design_name"], case_index
                        )
                    )
        records.append(
            {
                "design": info["design_name"],
                "nodes": graph.num_nodes(),
                "intra_gate_edges": graph.num_edges("intra_gate"),
                "intra_module_edges": graph.num_edges("intra_module"),
                "pis": pi_count,
                "pos": po_count,
                "cases": len(cases),
                "labeled_endpoint_samples": sum(labeled_counts),
                "missing_endpoint_samples": sum(missing_counts),
                "case_labeled_min": min(labeled_counts),
                "case_labeled_mean": sum(labeled_counts) / len(labeled_counts),
                "case_labeled_max": max(labeled_counts),
                "ntype_dim": int(graph.ndata["ntype"].shape[1]),
                "gate_type_dim": int(graph.ndata["ntype_gate"].shape[1]),
                "module_type_dim": int(graph.ndata["ntype_module"].shape[1]),
            }
        )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
