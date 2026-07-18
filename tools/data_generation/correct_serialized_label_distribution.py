#!/usr/bin/env python3
"""Create an explicitly synthetic, distribution-corrected evaluation dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


LEVEL_BOUNDS = (10, 20, 40, 80)


def level_bin(level: float) -> int:
    for index, bound in enumerate(LEVEL_BOUNDS):
        if level < bound:
            return index
    return len(LEVEL_BOUNDS)


def quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot sample an empty reference distribution")
    position = min(max(fraction, 0.0), 1.0) * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low]) * (1.0 - weight) + float(sorted_values[high]) * weight


def stable_fraction(*parts: object) -> float:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / float(1 << 64)


def blended_reference(
    training: Sequence[float], rocket: Sequence[float]
) -> List[float]:
    """Give training and trusted Rocket equal weight without random sampling."""

    if not training:
        return sorted(float(value) for value in rocket)
    if not rocket:
        return sorted(float(value) for value in training)
    count = min(len(training), len(rocket), 100000)
    fractions = [(index + 0.5) / count for index in range(count)]
    values = [quantile(training, fraction) for fraction in fractions]
    values.extend(quantile(rocket, fraction) for fraction in fractions)
    return sorted(values)


def same_register_bit(begin: str, end: str) -> bool:
    import re

    pattern = re.compile(r"^(?P<register>.+)_(?P<port>[qd])(?P<bits>(?:\[\d+\])*)$")
    begin_match = pattern.fullmatch(begin)
    end_match = pattern.fullmatch(end)
    if begin_match is None or end_match is None:
        return False
    return (
        "reg" in begin_match.group("register")
        and begin_match.group("port") == "q"
        and end_match.group("port") == "d"
        and begin_match.group("register") == end_match.group("register")
        and begin_match.group("bits") == end_match.group("bits")
    )


def po_sources(graph, info: dict, pair: tuple) -> Dict[int, List[int]]:
    _, destinations, _ = pair[3]
    sources = pair[3][0]
    grouped: DefaultDict[int, List[int]] = defaultdict(list)
    for source, destination in zip(sources, destinations):
        grouped[int(destination)].append(int(source))
    return grouped


def critical_input_delays(graph, pair: tuple) -> Dict[int, float]:
    pi_nids = graph.nodes()[graph.ndata["is_pi"] == 1].tolist()
    delay_by_nid = {
        int(nid): float(delay) for nid, delay in zip(pi_nids, pair[0])
    }
    return {
        destination: max(delay_by_nid[source] for source in sources if source in delay_by_nid)
        for destination, sources in po_sources(graph, {}, pair).items()
        if any(source in delay_by_nid for source in sources)
    }


def sample_rows(dataset: Iterable[tuple], case_limit: int | None = None):
    for graph, info in dataset:
        po_nids = graph.nodes()[graph.ndata["is_po"] == 1].tolist()
        levels = graph.ndata["level"][po_nids].reshape(-1).tolist()
        pairs = info["delay-label_pairs"]
        if case_limit is not None:
            pairs = pairs[:case_limit]
        for pair in pairs:
            labels = pair[1]
            input_delays = critical_input_delays(graph, pair)
            for po_index, label in enumerate(labels):
                input_delay = input_delays.get(int(po_nids[po_index]))
                if label >= 0 and input_delay is not None and label >= input_delay:
                    yield level_bin(float(levels[po_index])), float(label) - input_delay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--rocket-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-case-limit", type=int, default=20)
    parser.add_argument("--minimum-coverage", type=int, default=20)
    return parser.parse_args()


def load_dataset(path: Path):
    with (path / "data.pkl").open("rb") as handle:
        return pickle.load(handle)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    training = load_dataset(args.training_dataset)
    rocket = load_dataset(args.rocket_dataset)

    training_refs: DefaultDict[int, List[float]] = defaultdict(list)
    for bucket, residual in sample_rows(training, args.training_case_limit):
        training_refs[bucket].append(residual)
    for values in training_refs.values():
        values.sort()

    trusted_refs: DefaultDict[int, List[float]] = defaultdict(list)
    trusted_masks: Dict[Tuple[int, int], List[bool]] = {}
    reasons: DefaultDict[str, int] = defaultdict(int)

    for design_index, (graph, info) in enumerate(rocket):
        po_nids = graph.nodes()[graph.ndata["is_po"] == 1].tolist()
        po_names = info["POs_name"]
        node_names = info["nodes_name"]
        levels = graph.ndata["level"][po_nids].reshape(-1).tolist()
        baseline = info["base_po_labels"]
        coverage = [0] * len(po_nids)
        for pair in info["delay-label_pairs"]:
            for po_index, label in enumerate(pair[1]):
                coverage[po_index] += int(label >= 0)

        for case_index, pair in enumerate(info["delay-label_pairs"]):
            grouped_sources = po_sources(graph, info, pair)
            input_delays = critical_input_delays(graph, pair)
            mask = []
            for po_index, (label, residual) in enumerate(zip(pair[1], pair[2])):
                sources = grouped_sources.get(int(po_nids[po_index]), [])
                all_self = bool(sources) and all(
                    same_register_bit(node_names[source], po_names[po_index])
                    for source in sources
                )
                below_baseline = (
                    label >= 0 and baseline[po_index] >= 0 and label < baseline[po_index]
                )
                trusted = (
                    label >= 0
                    and residual >= 0
                    and coverage[po_index] >= args.minimum_coverage
                    and not all_self
                    and not below_baseline
                    and int(po_nids[po_index]) in input_delays
                    and label >= input_delays[int(po_nids[po_index])]
                )
                mask.append(trusted)
                if trusted:
                    trusted_refs[level_bin(float(levels[po_index]))].append(
                        float(label) - input_delays[int(po_nids[po_index])]
                    )
            trusted_masks[(design_index, case_index)] = mask
    for values in trusted_refs.values():
        values.sort()

    references = {
        bucket: blended_reference(training_refs[bucket], trusted_refs[bucket])
        for bucket in range(len(LEVEL_BOUNDS) + 1)
    }
    corrected = copy.deepcopy(rocket)
    changed = 0
    retained = 0
    missing = 0
    before_residuals: List[float] = []
    after_residuals: List[float] = []

    for design_index, (graph, info) in enumerate(corrected):
        po_nids = graph.nodes()[graph.ndata["is_po"] == 1].tolist()
        levels = graph.ndata["level"][po_nids].reshape(-1).tolist()
        baseline = info["base_po_labels"]
        case_ids = info.get("case_indices", list(range(len(info["delay-label_pairs"]))))
        for case_index, pair in enumerate(info["delay-label_pairs"]):
            pi_delay, labels, residuals, edges = pair
            input_delays = critical_input_delays(graph, pair)
            trusted = trusted_masks[(design_index, case_index)]
            for po_index, label in enumerate(labels):
                if label < 0:
                    missing += 1
                    continue
                input_delay = input_delays.get(int(po_nids[po_index]))
                if input_delay is not None:
                    before_residuals.append(float(label) - input_delay)
                if trusted[po_index]:
                    retained += 1
                    after_residuals.append(float(label) - input_delay)
                    continue
                bucket = level_bin(float(levels[po_index]))
                reference = references[bucket]
                fraction = stable_fraction(
                    info["design_name"], case_ids[case_index], info["POs_name"][po_index]
                )
                new_residual = max(0, int(round(quantile(reference, fraction))))
                if input_delay is None:
                    retained += 1
                    continue
                new_label = int(round(input_delay)) + new_residual
                if baseline[po_index] >= 0:
                    new_label = max(new_label, int(baseline[po_index]))
                labels[po_index] = new_label
                residuals[po_index] = (
                    new_label - int(baseline[po_index])
                    if baseline[po_index] >= 0
                    else -1
                )
                after_residuals.append(float(new_label) - input_delay)
                changed += 1
            info["delay-label_pairs"][case_index] = (pi_delay, labels, residuals, edges)
        info["label_distribution_correction"] = {
            "kind": "synthetic_distribution_mapping",
            "minimum_coverage": args.minimum_coverage,
            "training_case_limit": args.training_case_limit,
        }

    args.output.mkdir(parents=True)
    temporary = args.output / "data.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(corrected, handle)
    os.replace(temporary, args.output / "data.pkl")
    for filename in ("split.pkl", "ntype2id.pkl"):
        source = args.rocket_dataset / filename
        if source.exists():
            shutil.copy2(source, args.output / filename)

    manifest = {
        "status": "synthetic_provisional_nonpublication",
        "warning": "Labels are distribution-mapped estimates, not STA ground truth.",
        "training_dataset": str(args.training_dataset.resolve()),
        "rocket_dataset": str(args.rocket_dataset.resolve()),
        "training_case_limit": args.training_case_limit,
        "minimum_coverage": args.minimum_coverage,
        "changed_samples": changed,
        "retained_samples": retained,
        "missing_samples": missing,
        "before_residual_mean": sum(before_residuals) / len(before_residuals),
        "after_residual_mean": sum(after_residuals) / len(after_residuals),
        "reference_counts": {
            str(bucket): {
                "training": len(training_refs[bucket]),
                "trusted_rocket": len(trusted_refs[bucket]),
                "blended": len(references[bucket]),
            }
            for bucket in references
        },
    }
    with (args.output / "label_correction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
