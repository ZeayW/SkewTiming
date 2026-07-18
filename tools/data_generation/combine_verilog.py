#!/usr/bin/env python3
"""Combine ordered Verilog sources into one self-contained input file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        for source in args.sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output)
            output.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
