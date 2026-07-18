#!/usr/bin/env python3
"""Rebuild serialized graph type one-hots with a canonical training mapping."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import torch as th


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ntype-file", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)

    with args.ntype_file.open("rb") as handle:
        ntype2id, ntype2id_gate, ntype2id_module = pickle.load(handle)
    with (args.input_dir / "data.pkl").open("rb") as handle:
        dataset = pickle.load(handle)

    for graph, graph_info in dataset:
        is_module = graph.ndata["is_module"].tolist()
        node_types = graph_info["ntype"]
        ntype = th.zeros((graph.num_nodes(), len(ntype2id)), dtype=th.float)
        ntype_gate = th.zeros((graph.num_nodes(), len(ntype2id_gate)), dtype=th.float)
        ntype_module = th.zeros(
            (graph.num_nodes(), len(ntype2id_module)), dtype=th.float
        )
        for node_id, node_type in enumerate(node_types):
            ntype[node_id, ntype2id[node_type]] = 1
            if is_module[node_id]:
                ntype_module[node_id, ntype2id_module[node_type]] = 1
            else:
                ntype_gate[node_id, ntype2id_gate[node_type]] = 1
        graph.ndata["ntype"] = ntype
        graph.ndata["ntype_gate"] = ntype_gate
        graph.ndata["ntype_module"] = ntype_module

    with tempfile.NamedTemporaryFile(dir=str(args.output_dir), delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(dataset, handle)
    os.replace(str(temporary), str(args.output_dir / "data.pkl"))
    shutil.copy2(args.input_dir / "split.pkl", args.output_dir / "split.pkl")
    shutil.copy2(args.ntype_file, args.output_dir.parent / "ntype2id.pkl")
    print(
        "re-encoded {} graphs with dimensions {}/{}/{}".format(
            len(dataset), len(ntype2id), len(ntype2id_gate), len(ntype2id_module)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
