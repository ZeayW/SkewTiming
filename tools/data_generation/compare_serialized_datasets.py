#!/usr/bin/env python3
"""Compare two serialized NUA-Timer datasets for exact semantic equality."""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch


def compare_value(path, left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError('{} tensor type mismatch'.format(path))
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(
                '{} tensor metadata mismatch: {}/{} vs {}/{}'.format(
                    path, left.dtype, tuple(left.shape), right.dtype, tuple(right.shape)
                )
            )
        if not torch.equal(left, right):
            mismatch = torch.nonzero(left != right, as_tuple=False)
            location = mismatch[0].tolist() if mismatch.numel() else []
            raise AssertionError('{} tensor differs at {}'.format(path, location))
        return

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise AssertionError('{} ndarray type mismatch'.format(path))
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError('{} ndarray metadata mismatch'.format(path))
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError('{} ndarray differs'.format(path))
        return

    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise AssertionError('{} mapping type mismatch'.format(path))
        if set(left) != set(right):
            raise AssertionError(
                '{} keys differ: left_only={}, right_only={}'.format(
                    path, sorted(set(left) - set(right)), sorted(set(right) - set(left))
                )
            )
        for key in left:
            compare_value('{}[{!r}]'.format(path, key), left[key], right[key])
        return

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right):
            raise AssertionError('{} sequence type mismatch'.format(path))
        if len(left) != len(right):
            raise AssertionError(
                '{} length differs: {} vs {}'.format(path, len(left), len(right))
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_value('{}[{}]'.format(path, index), left_item, right_item)
        return

    if isinstance(left, float) and isinstance(right, float):
        if left == right or (math.isnan(left) and math.isnan(right)):
            return
        raise AssertionError('{} differs: {} vs {}'.format(path, left, right))

    if type(left) is not type(right) or left != right:
        raise AssertionError('{} differs: {!r} vs {!r}'.format(path, left, right))


def compare_graph(path, left, right):
    if left.ntypes != right.ntypes or left.canonical_etypes != right.canonical_etypes:
        raise AssertionError('{} graph schema differs'.format(path))
    if left.num_nodes() != right.num_nodes():
        raise AssertionError('{} node count differs'.format(path))

    if set(left.ndata) != set(right.ndata):
        raise AssertionError('{} node data keys differ'.format(path))
    for key in left.ndata:
        compare_value(
            '{}.ndata[{!r}]'.format(path, key), left.ndata[key], right.ndata[key]
        )

    for etype in left.canonical_etypes:
        if left.num_edges(etype=etype) != right.num_edges(etype=etype):
            raise AssertionError('{} {} edge count differs'.format(path, etype))
        left_edges = left.edges(etype=etype, form='all', order='eid')
        right_edges = right.edges(etype=etype, form='all', order='eid')
        compare_value('{} {} edges'.format(path, etype), left_edges, right_edges)
        left_data = left.edges[etype].data
        right_data = right.edges[etype].data
        if set(left_data) != set(right_data):
            raise AssertionError('{} {} edge data keys differ'.format(path, etype))
        for key in left_data:
            compare_value(
                '{} {}.edata[{!r}]'.format(path, etype, key),
                left_data[key],
                right_data[key],
            )


def load_dataset(path):
    with (path / 'data.pkl').open('rb') as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()

    left_dataset = load_dataset(args.left)
    right_dataset = load_dataset(args.right)
    if len(left_dataset) != len(right_dataset):
        raise AssertionError(
            'dataset length differs: {} vs {}'.format(
                len(left_dataset), len(right_dataset)
            )
        )

    total_cases = 0
    for index, ((left_graph, left_info), (right_graph, right_info)) in enumerate(
            zip(left_dataset, right_dataset)):
        left_name = left_info.get('design_name')
        right_name = right_info.get('design_name')
        if left_name != right_name:
            raise AssertionError(
                'dataset[{}] design differs: {} vs {}'.format(
                    index, left_name, right_name
                )
            )
        prefix = 'dataset[{}] {}'.format(index, left_name)
        compare_graph(prefix, left_graph, right_graph)
        compare_value(prefix + '.graph_info', left_info, right_info)
        total_cases += len(left_info.get('delay-label_pairs', []))

    print(
        'Datasets are exactly equivalent: designs={}, cases={}'.format(
            len(left_dataset), total_cases
        )
    )


if __name__ == '__main__':
    main()
