#!/usr/bin/env python3
import argparse
from pathlib import Path


REG_DECLARATION = "  reg [25:0] lce_resp_o;"
WIRE_DECLARATION = "  wire [25:0] lce_resp_o;"
CONSTANT_ASSIGN_ANCHOR = "  assign lce_resp_o[23] = 1'b1;"
CONSTANT_ASSIGN = "  assign lce_resp_o[25] = 1'b0;"
PROCEDURAL_ASSIGN = (
    "    if(1'b1) begin\n"
    "      { lce_resp_o[25:25] } <= { 1'b0 };\n"
    "    end\n"
)
ICACHE_REG_DECLARATION = "  reg [95:0] icache_pc_gen_data_o;"
ICACHE_WIRE_AND_REG_DECLARATION = (
    "  wire [95:0] icache_pc_gen_data_o;\n"
    "  reg [63:0] icache_pc_gen_data_low;"
)
ICACHE_UPPER_ASSIGN = "  assign icache_pc_gen_data_o[95:64] ="
ICACHE_LOWER_ASSIGN = (
    "  assign icache_pc_gen_data_o[63:0] = icache_pc_gen_data_low;"
)
ICACHE_PROCEDURAL_ASSIGN = (
    "      { icache_pc_gen_data_o[63:0] } <= { eaddr_tl_r[63:0] };"
)
ICACHE_REPLACEMENT_ASSIGN = (
    "      { icache_pc_gen_data_low[63:0] } <= { eaddr_tl_r[63:0] };"
)


def patch_bp_multi_top(text: str) -> str:
    expected_counts = {
        REG_DECLARATION: 1,
        PROCEDURAL_ASSIGN: 1,
        ICACHE_REG_DECLARATION: 1,
        ICACHE_UPPER_ASSIGN: 1,
        ICACHE_PROCEDURAL_ASSIGN: 1,
    }
    for fragment, expected in expected_counts.items():
        actual = text.count(fragment)
        if actual != expected:
            raise ValueError(
                f"expected {expected} occurrence of {fragment!r}, found {actual}"
            )

    before_declaration, declaration, after_declaration = text.partition(REG_DECLARATION)
    if not declaration or CONSTANT_ASSIGN_ANCHOR not in after_declaration:
        raise ValueError("constant assignment anchor not found after register declaration")

    after_declaration = after_declaration.replace(
        CONSTANT_ASSIGN_ANCHOR,
        f"{CONSTANT_ASSIGN}\n{CONSTANT_ASSIGN_ANCHOR}",
        1,
    )
    text = before_declaration + WIRE_DECLARATION + after_declaration
    text = text.replace(PROCEDURAL_ASSIGN, "", 1)
    text = text.replace(
        ICACHE_REG_DECLARATION, ICACHE_WIRE_AND_REG_DECLARATION, 1
    )
    text = text.replace(
        ICACHE_UPPER_ASSIGN,
        f"{ICACHE_LOWER_ASSIGN}\n{ICACHE_UPPER_ASSIGN}",
        1,
    )
    return text.replace(
        ICACHE_PROCEDURAL_ASSIGN, ICACHE_REPLACEMENT_ASSIGN, 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the bp_multi_top pickled netlist compatible with TangDynasty."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.source.resolve() == args.destination.resolve():
        raise SystemExit("source and destination must differ")

    source_text = args.source.read_text(encoding="utf-8")
    patched_text = patch_bp_multi_top(source_text)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(patched_text, encoding="utf-8")
    print(f"wrote {args.destination} ({len(patched_text)} bytes)")


if __name__ == "__main__":
    main()
