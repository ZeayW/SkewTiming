#!/usr/bin/env python3
"""Audit large structural netlists with the commercial-netlist frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from commercial_netlist_parser import CommercialNetlistParser  # noqa: E402


def audit_netlist(
    netlist: Path,
    top_module: Optional[str] = None,
    mode: str = "stats",
    strict: bool = True,
    cell_port_json: Optional[Path] = None,
    reachability: bool = False,
) -> dict:
    parser_kwargs = {
        "top_module": top_module,
        "mode": mode,
        "strict": strict,
    }
    if cell_port_json is None:
        parser = CommercialNetlistParser(netlist, **parser_kwargs)
    else:
        parser = CommercialNetlistParser.from_cell_port_json(
            netlist, cell_port_json, **parser_kwargs
        )
    result = parser.parse()
    record = result.summary(include_reachability=reachability)
    record["design"] = netlist.stem
    record["endpoint_definition"] = "selected_module_declared_output_bits"
    return record


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Parse the selected top module of commercial Verilog netlists and "
            "report structural node/endpoint counts without timing labels"
        )
    )
    argument_parser.add_argument(
        "--netlist", type=Path, action="append", required=True,
        help="Verilog netlist; repeat for multiple designs",
    )
    argument_parser.add_argument(
        "--top-module",
        help="module to parse; defaults to the first module in each netlist",
    )
    argument_parser.add_argument(
        "--mode", choices=("stats", "graph"), default="stats",
        help="stats is bounded-memory; graph materialises compact bit-level edges",
    )
    argument_parser.add_argument(
        "--cell-port-json", type=Path,
        help="optional vendor cell-port direction map",
    )
    argument_parser.add_argument(
        "--permissive", action="store_true",
        help="record unsupported selected-module statements instead of failing",
    )
    argument_parser.add_argument(
        "--reachability", action="store_true",
        help="in graph mode, count the transitive fanin of declared outputs",
    )
    argument_parser.add_argument(
        "--output-json", type=Path,
        help="optional JSON output path; stdout is always written",
    )
    args = argument_parser.parse_args()
    if args.reachability and args.mode != "graph":
        argument_parser.error("--reachability requires --mode graph")

    records = [
        audit_netlist(
            path.resolve(),
            top_module=args.top_module,
            mode=args.mode,
            strict=not args.permissive,
            cell_port_json=args.cell_port_json,
            reachability=args.reachability,
        )
        for path in args.netlist
    ]
    payload = json.dumps(records, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
