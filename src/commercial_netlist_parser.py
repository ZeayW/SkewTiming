"""Streaming structural parser for large commercial Verilog netlists.

The historical NUA-Timer parser assumes that every semicolon-delimited item in
the file belongs to one flat module.  Commercial tools commonly append helper
module definitions (RAMs, muxes, arithmetic operators, and simulation models)
after the generated top-level netlist.  Parsing those helper bodies as top-level
logic corrupts the graph and makes otherwise valid netlists fail.

This module provides a dependency-free frontend with two deliberate modes:

``stats``
    Parse and validate the selected module without materialising graph edges.
    This is suitable for multi-GB netlists and reports declared node/endpoint
    counts with bounded memory use.

``graph``
    Materialise a compact bit-level dependency graph.  Edges use integer arrays
    rather than Python objects so the result remains practical for large
    generated designs.

The parser is structural: procedural blocks in the selected top module are
reported as unsupported.  Procedural code in appended helper modules is never
visited, because helper modules are library definitions rather than instances
of the selected design.
"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


_IDENTIFIER = r"(?:\\[^\s,(){};]+|[A-Za-z_$][A-Za-z0-9_$]*)"
_IDENTIFIER_RE = re.compile(_IDENTIFIER)
_REFERENCE_RE = re.compile(
    r"(?P<name>{})(?:\s*\[\s*(?P<select>[^\]]+)\s*\])?".format(_IDENTIFIER)
)
_SIMPLE_REFERENCE_RE = re.compile(
    r"^(?P<name>{})(?:\s*\[\s*(?P<select>[^\]]+)\s*\])?$".format(_IDENTIFIER)
)
_CONSTANT_RE = re.compile(
    r"(?:(?P<width>\d+)\s*)?'\s*(?P<base>[bBoOdDhH])\s*(?P<digits>[0-9a-fA-FxXzZ?_]+)"
)
_UNSIZED_NUMBER_RE = re.compile(r"^[-+]?\d+$")
_DECLARATION_RE = re.compile(
    r"^(?P<kind>input|output|inout|wire|tri|tri0|tri1|wand|wor|logic|reg|supply0|supply1)\b"
    r"(?P<body>.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_STATIC_RANGE_RE = re.compile(r"^\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]")
_ATTRIBUTE_RE = re.compile(r"^\s*(?:\(\*.*?\*\)\s*)+", flags=re.DOTALL)
_MODULE_RE = re.compile(
    r"^module\s+(?:automatic\s+)?(?P<name>{})".format(_IDENTIFIER),
    flags=re.IGNORECASE,
)
_PROCEDURAL_RE = re.compile(
    r"^(?:always(?:_comb|_ff|_latch)?|initial|final|generate|endgenerate|for|if|case|endcase)\b",
    flags=re.IGNORECASE,
)

_DECLARATION_QUALIFIERS = {
    "wire", "reg", "logic", "signed", "unsigned", "var", "tri", "tri0",
    "tri1", "wand", "wor", "supply0", "supply1", "const",
}
_IGNORED_STATEMENT_PREFIXES = (
    "parameter", "localparam", "specparam", "genvar", "timeunit",
    "timeprecision", "default_nettype", "timescale",
)
_EXPRESSION_KEYWORDS = {
    "and", "or", "not", "xor", "xnor", "nand", "nor", "if", "else",
    "begin", "end", "case", "endcase", "default", "signed", "unsigned",
}
_PRIMITIVES = {
    "and", "nand", "or", "nor", "xor", "xnor", "buf", "not", "bufif0",
    "bufif1", "notif0", "notif1", "tran", "tranif0", "tranif1",
    "rtran", "rtranif0", "rtranif1", "cmos", "rcmos", "nmos", "pmos",
    "rnmos", "rpmos", "pullup", "pulldown",
}
_OUTPUT_PORT_RE = re.compile(
    r"^(?:o|out|output|y|z|zn|q|qn|qo|sum|co|cout|rd|rdata|dout)(?:\d+)?$",
    flags=re.IGNORECASE,
)
_SEQUENTIAL_CELL_RE = re.compile(
    r"(?:^|_)(?:dff|sdff|flop|latch|fd[rcpes]*|register)(?:_|$)",
    flags=re.IGNORECASE,
)


class CommercialNetlistError(ValueError):
    """Raised when the selected module is not structurally representable."""


@dataclass(frozen=True)
class Statement:
    text: str
    line: int


@dataclass
class ParseDiagnostics:
    counts: Counter = field(default_factory=Counter)
    samples: List[dict] = field(default_factory=list)
    max_samples: int = 32

    def record(self, code: str, message: str, line: Optional[int] = None) -> None:
        self.counts[code] += 1
        if len(self.samples) < self.max_samples:
            sample = {"code": code, "message": message}
            if line is not None:
                sample["line"] = line
            self.samples.append(sample)

    def as_dict(self) -> dict:
        return {
            "counts": dict(sorted(self.counts.items())),
            "samples": list(self.samples),
        }


@dataclass
class Signal:
    name: str
    kind: str
    left: Optional[int] = None
    right: Optional[int] = None

    @property
    def width(self) -> int:
        if self.left is None:
            return 1
        return abs(self.left - self.right) + 1

    def bit_names(self) -> List[str]:
        if self.left is None:
            return [self.name]
        step = 1 if self.right >= self.left else -1
        return [
            "{}[{}]".format(self.name, bit)
            for bit in range(self.left, self.right + step, step)
        ]


class SignalTable:
    """Base-name signal table that expands vectors only when needed."""

    _DIRECTION_PRIORITY = {
        "wire": 0, "tri": 0, "tri0": 0, "tri1": 0, "wand": 0, "wor": 0,
        "logic": 0, "reg": 0, "supply0": 0, "supply1": 0,
        "input": 2, "output": 2, "inout": 3,
    }

    def __init__(self, diagnostics: ParseDiagnostics) -> None:
        self._signals: Dict[str, Signal] = {}
        self.diagnostics = diagnostics
        self.declared_bits = 0

    def __contains__(self, name: str) -> bool:
        return _normalise_identifier(name) in self._signals

    def values(self) -> Iterable[Signal]:
        return self._signals.values()

    def get(self, name: str) -> Optional[Signal]:
        return self._signals.get(_normalise_identifier(name))

    def declare(
        self,
        name: str,
        kind: str,
        bit_range: Optional[Tuple[int, int]],
        line: int,
    ) -> Signal:
        name = _normalise_identifier(name)
        kind = kind.lower()
        left, right = bit_range if bit_range is not None else (None, None)
        incoming = Signal(name, kind, left, right)
        current = self._signals.get(name)
        if current is None:
            self._signals[name] = incoming
            self.declared_bits += incoming.width
            return incoming

        if current.width != incoming.width:
            raise CommercialNetlistError(
                "conflicting widths for {} at line {}: {} versus {}".format(
                    name, line, current.width, incoming.width
                )
            )
        if self._DIRECTION_PRIORITY.get(kind, 0) > self._DIRECTION_PRIORITY.get(
            current.kind, 0
        ):
            current.kind = kind
        return current

    def implicit(self, name: str, line: int) -> Signal:
        name = _normalise_identifier(name)
        signal = self._signals.get(name)
        if signal is None:
            signal = Signal(name, "wire")
            self._signals[name] = signal
            self.declared_bits += 1
            self.diagnostics.record(
                "implicit_scalar_net", "created implicit net {}".format(name), line
            )
        return signal


class CompactStructuralGraph:
    """Compact bit-level graph produced by the commercial frontend."""

    def __init__(self) -> None:
        self.node_names: List[str] = []
        self.node_types: List[str] = []
        self.name_to_id: Dict[str, int] = {}
        self.is_input = bytearray()
        self.is_output = bytearray()
        self.is_module = bytearray()
        self.edge_sources = array("Q")
        self.edge_destinations = array("Q")
        self.edge_is_module = bytearray()
        self.edge_bit_positions = array("i")

    def add_node(
        self,
        name: str,
        node_type: str = "wire",
        is_input: bool = False,
        is_output: bool = False,
        is_module: bool = False,
    ) -> int:
        node_id = self.name_to_id.get(name)
        if node_id is None:
            node_id = len(self.node_names)
            self.name_to_id[name] = node_id
            self.node_names.append(name)
            self.node_types.append(node_type)
            self.is_input.append(bool(is_input))
            self.is_output.append(bool(is_output))
            self.is_module.append(bool(is_module))
            return node_id

        if node_type != "wire":
            self.node_types[node_id] = node_type
        self.is_input[node_id] = self.is_input[node_id] or bool(is_input)
        self.is_output[node_id] = self.is_output[node_id] or bool(is_output)
        self.is_module[node_id] = self.is_module[node_id] or bool(is_module)
        return node_id

    def add_edge(
        self,
        source: int,
        destination: int,
        is_module: bool,
        bit_position: int = 0,
    ) -> None:
        self.edge_sources.append(source)
        self.edge_destinations.append(destination)
        self.edge_is_module.append(bool(is_module))
        self.edge_bit_positions.append(bit_position)

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return len(self.edge_sources)

    def backward_reachable_count(self) -> int:
        """Count nodes in the transitive fanin of structural outputs."""
        incoming: List[List[int]] = [[] for _ in range(self.num_nodes)]
        for source, destination in zip(self.edge_sources, self.edge_destinations):
            incoming[destination].append(source)
        pending = [index for index, flag in enumerate(self.is_output) if flag]
        seen = set(pending)
        while pending:
            destination = pending.pop()
            for source in incoming[destination]:
                if source not in seen:
                    seen.add(source)
                    pending.append(source)
        return len(seen)


@dataclass
class CommercialParseResult:
    netlist: Path
    module_name: str
    mode: str
    signals: SignalTable
    graph: Optional[CompactStructuralGraph]
    diagnostics: ParseDiagnostics
    statement_count: int
    assignment_count: int
    instance_count: int

    @property
    def input_count(self) -> int:
        return sum(
            signal.width for signal in self.signals.values()
            if signal.kind in ("input", "inout")
        )

    @property
    def endpoint_count(self) -> int:
        return sum(
            signal.width for signal in self.signals.values()
            if signal.kind in ("output", "inout")
        )

    def summary(self, include_reachability: bool = False) -> dict:
        graph_nodes = None if self.graph is None else self.graph.num_nodes
        graph_edges = None if self.graph is None else self.graph.num_edges
        reachable = None
        if include_reachability and self.graph is not None:
            reachable = self.graph.backward_reachable_count()
        return {
            "parser": "commercial_structural_v1",
            "netlist": str(self.netlist.resolve()),
            "module": self.module_name,
            "mode": self.mode,
            "declared_nodes": self.signals.declared_bits,
            "pre_filter_nodes": (
                self.signals.declared_bits if graph_nodes is None else graph_nodes
            ),
            "pre_filter_endpoints": self.endpoint_count,
            "pis": self.input_count,
            "graph_edges": graph_edges,
            "post_filter_nodes": reachable,
            "statements": self.statement_count,
            "assignments": self.assignment_count,
            "instances": self.instance_count,
            "diagnostics": self.diagnostics.as_dict(),
        }


def _normalise_identifier(name: str) -> str:
    name = name.strip()
    if name.startswith("\\"):
        name = name[1:].rstrip()
    return name


def _remove_comments(line: str, in_block_comment: bool) -> Tuple[str, bool]:
    if not in_block_comment and "//" not in line and "/*" not in line:
        return line, False
    result: List[str] = []
    index = 0
    quoted = False
    escaped = False
    while index < len(line):
        if in_block_comment:
            close = line.find("*/", index)
            if close < 0:
                return "".join(result), True
            index = close + 2
            in_block_comment = False
            continue
        char = line[index]
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
            result.append(char)
            index += 1
            continue
        if line.startswith("//", index):
            break
        if line.startswith("/*", index):
            in_block_comment = True
            index += 2
            continue
        result.append(char)
        index += 1
    return "".join(result), in_block_comment


def iter_verilog_statements(path: Path) -> Iterator[Statement]:
    """Yield semicolon statements and standalone ``endmodule`` tokens."""
    buffer: List[str] = []
    start_line: Optional[int] = None
    in_block_comment = False
    paren = bracket = brace = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line, in_block_comment = _remove_comments(raw_line, in_block_comment)
            if start_line is None and line.lstrip().startswith("`"):
                buffer = []
                start_line = None
                continue

            # Commercial netlists overwhelmingly use one statement per line.
            # Handle that form with C-level string counts; only unusual lines
            # (multiple semicolons or inline endmodule) need character scanning.
            stripped = line.lstrip()
            inline_endmodule = "endmodule" in line
            semicolon_count = line.count(";")
            if semicolon_count <= 1 and not inline_endmodule:
                if start_line is None and stripped:
                    start_line = line_number
                if semicolon_count == 0:
                    buffer.append(line)
                    paren += line.count("(") - line.count(")")
                    bracket += line.count("[") - line.count("]")
                    brace += line.count("{") - line.count("}")
                    continue

                semicolon = line.index(";")
                before = line[:semicolon]
                after = line[semicolon + 1:]
                buffer.append(before)
                paren += before.count("(") - before.count(")")
                bracket += before.count("[") - before.count("]")
                brace += before.count("{") - before.count("}")
                if not (paren or bracket or brace):
                    text = "".join(buffer).strip()
                    if text:
                        yield Statement(text, start_line or line_number)
                    buffer = []
                    start_line = None
                    if after.strip():
                        buffer.append(after)
                        start_line = line_number
                        paren += after.count("(") - after.count(")")
                        bracket += after.count("[") - after.count("]")
                        brace += after.count("{") - after.count("}")
                else:
                    buffer.append(";" + after)
                    paren += after.count("(") - after.count(")")
                    bracket += after.count("[") - after.count("]")
                    brace += after.count("{") - after.count("}")
                continue

            index = 0
            quoted = False
            escaped = False
            while index < len(line):
                if (
                    not quoted
                    and not (paren or bracket or brace)
                    and line.startswith("endmodule", index)
                    and (index == 0 or not (line[index - 1].isalnum() or line[index - 1] in "_$"))
                    and (
                        index + 9 == len(line)
                        or not (
                            line[index + 9].isalnum()
                            or line[index + 9] in "_$"
                        )
                    )
                ):
                    prefix = "".join(buffer).strip()
                    if prefix:
                        yield Statement(prefix, start_line or line_number)
                    buffer = []
                    start_line = None
                    yield Statement("endmodule", line_number)
                    index += len("endmodule")
                    continue

                char = line[index]
                if start_line is None and not char.isspace():
                    start_line = line_number
                buffer.append(char)
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                    index += 1
                    continue
                if char == '"':
                    quoted = True
                elif char == "(":
                    paren += 1
                elif char == ")":
                    paren = max(0, paren - 1)
                elif char == "[":
                    bracket += 1
                elif char == "]":
                    bracket = max(0, bracket - 1)
                elif char == "{":
                    brace += 1
                elif char == "}":
                    brace = max(0, brace - 1)
                elif char == ";" and not (paren or bracket or brace):
                    text = "".join(buffer[:-1]).strip()
                    if text:
                        yield Statement(text, start_line or line_number)
                    buffer = []
                    start_line = None
                index += 1

    trailing = "".join(buffer).strip()
    if trailing:
        yield Statement(trailing, start_line or 1)


def split_top_level(text: str, delimiter: str = ",") -> List[str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    parts: List[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                raise CommercialNetlistError("unbalanced expression {!r}".format(text))
        elif char == delimiter and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if stack:
        raise CommercialNetlistError("unbalanced expression {!r}".format(text))
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _strip_top_level_initializer(text: str) -> str:
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "=" and depth == 0:
            return text[:index].strip()
    return text.strip()


def _parse_declaration(
    statement: str,
    line: int,
    signals: SignalTable,
    diagnostics: ParseDiagnostics,
) -> bool:
    match = _DECLARATION_RE.match(statement.strip())
    if match is None:
        return False
    kind = match.group("kind").lower()
    body = match.group("body").strip()
    tokens = body.split()
    while tokens and tokens[0].lower() in _DECLARATION_QUALIFIERS:
        tokens.pop(0)
    body = " ".join(tokens)

    bit_range = None
    range_match = _STATIC_RANGE_RE.match(body)
    if range_match is not None:
        bit_range = (int(range_match.group(1)), int(range_match.group(2)))
        body = body[range_match.end():].strip()
    elif body.startswith("["):
        diagnostics.record(
            "symbolic_packed_range",
            "treated non-static packed range as scalar: {}".format(body[:80]),
            line,
        )

    for item in split_top_level(body):
        item = _strip_top_level_initializer(item)
        identifier = _IDENTIFIER_RE.match(item.strip())
        if identifier is None:
            diagnostics.record(
                "unparsed_declaration_item", "could not parse {!r}".format(item), line
            )
            continue
        signals.declare(identifier.group(0), kind, bit_range, line)
    return True


def _parse_module_name(statement: str) -> Optional[str]:
    match = _MODULE_RE.match(statement.strip())
    if match is None:
        return None
    return _normalise_identifier(match.group("name"))


def _parse_ansi_module_ports(
    statement: str,
    line: int,
    signals: SignalTable,
    diagnostics: ParseDiagnostics,
) -> None:
    """Parse ANSI-style directions embedded in a module header."""
    module_match = _MODULE_RE.match(statement.strip())
    if module_match is None:
        return
    if re.search(r"\b(?:input|output|inout)\b", statement) is None:
        return
    text = statement.strip()
    index = module_match.end()
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "#":
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        index = _skip_balanced(text, index)
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "(":
        return
    close = _skip_balanced(text, index)
    body = text[index + 1:close - 1]

    current_kind = None
    current_range = None
    for part in split_top_level(body):
        direction = re.match(
            r"^(input|output|inout)\b(.*)$", part.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if direction is not None:
            current_kind = direction.group(1).lower()
            remainder = direction.group(2).strip()
            tokens = remainder.split()
            while tokens and tokens[0].lower() in _DECLARATION_QUALIFIERS:
                tokens.pop(0)
            remainder = " ".join(tokens)
            range_match = _STATIC_RANGE_RE.match(remainder)
            current_range = None
            if range_match is not None:
                current_range = (
                    int(range_match.group(1)), int(range_match.group(2))
                )
                remainder = remainder[range_match.end():].strip()
            part = remainder
        elif current_kind is None:
            # A non-ANSI header only lists names; body declarations carry the
            # directions and widths.
            continue

        identifier = _IDENTIFIER_RE.match(_strip_top_level_initializer(part))
        if identifier is None:
            diagnostics.record(
                "unparsed_ansi_port", "could not parse ANSI port {!r}".format(part), line
            )
            continue
        signals.declare(identifier.group(0), current_kind, current_range, line)


def _matching_outer_braces(text: str) -> bool:
    if not (text.startswith("{") and text.endswith("}")):
        return False
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def _parse_constant_bits(text: str) -> Optional[List[str]]:
    text = text.strip()
    match = _CONSTANT_RE.fullmatch(text)
    if match is None:
        if _UNSIZED_NUMBER_RE.fullmatch(text):
            return ["$const1" if int(text) else "$const0"]
        return None
    width_text, base, digits = match.group("width", "base", "digits")
    digits = digits.replace("_", "").lower()
    radix = {"b": 2, "o": 8, "d": 10, "h": 16}[base.lower()]
    width = int(width_text) if width_text is not None else None
    if any(char in digits for char in "xz?"):
        bit_count = width or max(1, len(digits) * (1 if radix == 2 else 4))
        return ["$constx"] * bit_count
    value = int(digits, radix)
    bit_count = width or max(1, value.bit_length())
    binary = format(value, "0{}b".format(bit_count))[-bit_count:]
    return ["$const1" if bit == "1" else "$const0" for bit in binary]


def _select_signal_bits(
    name: str,
    select: Optional[str],
    signals: SignalTable,
    diagnostics: ParseDiagnostics,
    line: int,
) -> List[str]:
    name = _normalise_identifier(name)
    signal = signals.get(name)
    if signal is None:
        signal = signals.implicit(name, line)
    if select is None:
        return signal.bit_names()
    select = select.strip()
    part = re.fullmatch(r"(-?\d+)\s*:\s*(-?\d+)", select)
    if part is not None:
        left, right = int(part.group(1)), int(part.group(2))
        step = 1 if right >= left else -1
        return ["{}[{}]".format(name, bit) for bit in range(left, right + step, step)]
    bit = re.fullmatch(r"-?\d+", select)
    if bit is not None:
        return ["{}[{}]".format(name, int(select))]
    diagnostics.record(
        "dynamic_select",
        "dynamic select {}[{}] depends on the full vector".format(name, select),
        line,
    )
    return signal.bit_names()


def expand_expression_bits(
    expression: str,
    signals: SignalTable,
    diagnostics: ParseDiagnostics,
    line: int,
) -> List[str]:
    """Return referenced bits in Verilog expression order."""
    expression = expression.strip()
    constant = _parse_constant_bits(expression)
    if constant is not None:
        return constant

    simple = _SIMPLE_REFERENCE_RE.fullmatch(expression)
    if simple is not None:
        return _select_signal_bits(
            simple.group("name"), simple.group("select"), signals, diagnostics, line
        )

    if _matching_outer_braces(expression):
        inner = expression[1:-1].strip()
        replication = re.fullmatch(r"(\d+)\s*\{(.*)\}", inner, flags=re.DOTALL)
        if replication is not None and _matching_outer_braces(
            "{" + replication.group(2) + "}"
        ):
            bits = expand_expression_bits(
                replication.group(2), signals, diagnostics, line
            )
            return bits * int(replication.group(1))
        result: List[str] = []
        for part in split_top_level(inner):
            result.extend(expand_expression_bits(part, signals, diagnostics, line))
        return result

    scrubbed = _CONSTANT_RE.sub(" ", expression)
    result = []
    seen = set()
    for match in _REFERENCE_RE.finditer(scrubbed):
        name = _normalise_identifier(match.group("name"))
        if name.lower() in _EXPRESSION_KEYWORDS:
            continue
        for bit_name in _select_signal_bits(
            name, match.group("select"), signals, diagnostics, line
        ):
            if bit_name not in seen:
                seen.add(bit_name)
                result.append(bit_name)
    return result


def _resize_bits(bits: Sequence[str], width: int) -> List[str]:
    bits = list(bits)
    if len(bits) < width:
        bits = ["$const0"] * (width - len(bits)) + bits
    elif len(bits) > width:
        bits = bits[-width:]
    return bits


def _split_top_level_ternary(text: str) -> Optional[Tuple[str, str, str]]:
    stack: List[str] = []
    question = None
    nested_questions = 0
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if stack:
                stack.pop()
        elif stack:
            continue
        elif char == "?":
            if question is None:
                question = index
            else:
                nested_questions += 1
        elif char == ":" and question is not None:
            if nested_questions:
                nested_questions -= 1
            else:
                return text[:question], text[question + 1:index], text[index + 1:]
    return None


def _classify_expression(expression: str) -> Tuple[str, bool]:
    if _SIMPLE_REFERENCE_RE.fullmatch(expression.strip()) or _parse_constant_bits(
        expression.strip()
    ) is not None:
        return "buf", False
    if _split_top_level_ternary(expression) is not None:
        return "mux", False
    operators = (
        (r"===|!==|==|!=", "eq", True),
        (r"<=|>=|<|>", "lt", True),
        (r"\*", "mult", True),
        (r"\+", "add", True),
        (r"-", "sub", True),
        (r">>|<<", "shift", True),
        (r"\^", "xor", False),
        (r"\|", "or", False),
        (r"&", "and", False),
        (r"~|!", "not", False),
    )
    for pattern, node_type, is_module in operators:
        if re.search(pattern, expression):
            return node_type, is_module
    return "expr", True


def _read_identifier(text: str, index: int) -> Tuple[Optional[str], int]:
    while index < len(text) and text[index].isspace():
        index += 1
    match = _IDENTIFIER_RE.match(text, index)
    if match is None:
        return None, index
    return _normalise_identifier(match.group(0)), match.end()


def _skip_balanced(text: str, index: int, opening: str = "(") -> int:
    closing = {
        "(": ")", "[": "]", "{": "}"}[opening]
    if index >= len(text) or text[index] != opening:
        raise CommercialNetlistError("expected {} in {!r}".format(opening, text[:120]))
    depth = 1
    index += 1
    while index < len(text) and depth:
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
        index += 1
    if depth:
        raise CommercialNetlistError("unterminated {} in {!r}".format(opening, text[:120]))
    return index


def _parse_instance(statement: str) -> Tuple[str, str, str]:
    index = 0
    cell_type, index = _read_identifier(statement, index)
    if cell_type is None:
        raise CommercialNetlistError("instance has no cell type: {!r}".format(statement[:160]))
    while index < len(statement) and statement[index].isspace():
        index += 1
    if index < len(statement) and statement[index] == "#":
        index += 1
        while index < len(statement) and statement[index].isspace():
            index += 1
        index = _skip_balanced(statement, index)
    instance_name, index = _read_identifier(statement, index)
    if instance_name is None:
        raise CommercialNetlistError(
            "{} instance has no name: {!r}".format(cell_type, statement[:160])
        )
    while index < len(statement) and statement[index].isspace():
        index += 1
    if index >= len(statement) or statement[index] != "(":
        raise CommercialNetlistError(
            "{} {} has no port list".format(cell_type, instance_name)
        )
    close = _skip_balanced(statement, index)
    if statement[close:].strip():
        raise CommercialNetlistError(
            "multiple instances per declaration are unsupported at {!r}".format(
                statement[close:close + 80]
            )
        )
    return cell_type, instance_name, statement[index + 1:close - 1]


def _parse_connections(port_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    parts = split_top_level(port_text)
    if not parts:
        return "positional", []
    if parts[0].lstrip().startswith("."):
        connections = []
        for part in parts:
            match = re.fullmatch(
                r"\.\s*({})\s*\((.*)\)".format(_IDENTIFIER),
                part,
                flags=re.DOTALL,
            )
            if match is None:
                raise CommercialNetlistError(
                    "invalid named connection {!r}".format(part[:160])
                )
            connections.append((_normalise_identifier(match.group(1)), match.group(2).strip()))
        return "named", connections
    return "positional", [(str(index), part) for index, part in enumerate(parts)]


def _cell_operation(cell_type: str) -> Tuple[str, bool]:
    lower = cell_type.lower()
    if lower in _PRIMITIVES:
        return lower, False
    if "mux" in lower:
        return "mux", False
    if lower.startswith(("add", "adder")):
        return "add", True
    if lower.startswith(("sub", "minus")):
        return "sub", True
    if lower.startswith(("mult", "mul")):
        return "mult", True
    if lower.startswith(("eq", "neq")):
        return "eq", True
    if lower.startswith(("lt", "le", "gt", "ge", "cmp")):
        return "lt", True
    if lower.startswith(("left", "right", "shift")):
        return "shift", True
    if lower.startswith(("read_port", "ram_")):
        return "memory", True
    return re.split(r"[_$]", lower, maxsplit=1)[0], True


def _is_output_port(cell_type: str, port: str) -> bool:
    lower_cell = cell_type.lower()
    lower_port = port.lower()
    if _OUTPUT_PORT_RE.fullmatch(lower_port):
        return True
    if lower_cell.startswith("clock_write_port_") and lower_port == "ram":
        return True
    if lower_cell.startswith(("read_port_", "ram_")) and lower_port.startswith("rd"):
        return True
    return False


class CommercialNetlistParser:
    """Parse one selected module from a commercial structural netlist."""

    def __init__(
        self,
        netlist: Union[Path, str],
        top_module: Optional[str] = None,
        mode: str = "stats",
        strict: bool = True,
        cell_ports: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> None:
        if mode not in ("stats", "graph"):
            raise ValueError("mode must be 'stats' or 'graph'")
        self.netlist = Path(netlist)
        self.top_module = top_module
        self.mode = mode
        self.strict = strict
        self.cell_ports = {
            cell.lower(): {port.lower(): direction.lower() for port, direction in ports.items()}
            for cell, ports in (cell_ports or {}).items()
        }
        self.diagnostics = ParseDiagnostics()
        self.signals = SignalTable(self.diagnostics)
        self.graph = CompactStructuralGraph() if mode == "graph" else None
        self._selected_module: Optional[str] = None
        self._inside_selected = False
        self._seen_selected = False
        self._statement_count = 0
        self._assignment_count = 0
        self._instance_count = 0

    @classmethod
    def from_cell_port_json(
        cls,
        netlist: Union[Path, str],
        cell_port_json: Union[Path, str],
        **kwargs,
    ) -> "CommercialNetlistParser":
        with Path(cell_port_json).open("r", encoding="utf-8") as handle:
            ports = json.load(handle)
        return cls(netlist, cell_ports=ports, **kwargs)

    def _fail_or_record(self, code: str, message: str, line: int) -> None:
        self.diagnostics.record(code, message, line)
        if self.strict:
            raise CommercialNetlistError("{}:{}: {}".format(self.netlist, line, message))

    def _materialise_declared_signals(self) -> None:
        if self.graph is None:
            return
        for signal in self.signals.values():
            for bit_name in signal.bit_names():
                self.graph.add_node(
                    bit_name,
                    node_type="input" if signal.kind in ("input", "inout") else "wire",
                    is_input=signal.kind in ("input", "inout"),
                    is_output=signal.kind in ("output", "inout"),
                )

    def _ensure_graph_node(self, name: str) -> Optional[int]:
        if self.graph is None:
            return None
        if name.startswith("$const"):
            return self.graph.add_node(name, node_type=name)
        base = name.split("[", 1)[0]
        signal = self.signals.get(base)
        return self.graph.add_node(
            name,
            node_type=(
                "input" if signal is not None and signal.kind in ("input", "inout")
                else "wire"
            ),
            is_input=signal is not None and signal.kind in ("input", "inout"),
            is_output=signal is not None and signal.kind in ("output", "inout"),
        )

    def _drive(
        self,
        destinations: Sequence[str],
        source_groups: Sequence[Sequence[str]],
        node_type: str,
        is_module: bool,
        line: int,
        triangular: bool = False,
    ) -> None:
        if self.graph is None:
            return
        for destination_index, destination_name in enumerate(destinations):
            destination = self._ensure_graph_node(destination_name)
            self.graph.node_types[destination] = node_type
            self.graph.is_module[destination] = bool(is_module)
            source_names = []
            for group in source_groups:
                if triangular and len(group) > 1:
                    source_names.extend(group[destination_index:])
                elif len(group) == len(destinations) and len(destinations) > 1:
                    source_names.append(group[destination_index])
                else:
                    source_names.extend(group)
            seen = set()
            bit_position = 1
            for source_name in source_names:
                if source_name in seen:
                    continue
                seen.add(source_name)
                source = self._ensure_graph_node(source_name)
                self.graph.add_edge(source, destination, is_module, bit_position)
                bit_position += 1

    def _parse_assign(self, statement: Statement) -> None:
        match = re.match(r"^assign\s+(.+?)\s*=\s*(.+)$", statement.text, flags=re.DOTALL)
        if match is None:
            self._fail_or_record(
                "invalid_assign", "invalid continuous assignment", statement.line
            )
            return
        lhs_text, rhs_text = match.groups()
        destinations = expand_expression_bits(
            lhs_text, self.signals, self.diagnostics, statement.line
        )
        if not destinations:
            self._fail_or_record(
                "empty_assign_lhs", "assignment has no destination bits", statement.line
            )
            return
        self._assignment_count += 1
        node_type, is_module = _classify_expression(rhs_text)
        ternary = _split_top_level_ternary(rhs_text)
        if ternary is not None:
            condition, true_value, false_value = ternary
            condition_bits = expand_expression_bits(
                condition, self.signals, self.diagnostics, statement.line
            )
            true_bits = _resize_bits(
                expand_expression_bits(
                    true_value, self.signals, self.diagnostics, statement.line
                ),
                len(destinations),
            )
            false_bits = _resize_bits(
                expand_expression_bits(
                    false_value, self.signals, self.diagnostics, statement.line
                ),
                len(destinations),
            )
            self._drive(
                destinations,
                [condition_bits, true_bits, false_bits],
                "mux",
                False,
                statement.line,
            )
            return

        source_bits = expand_expression_bits(
            rhs_text, self.signals, self.diagnostics, statement.line
        )
        if not source_bits:
            self._fail_or_record(
                "empty_assign_rhs",
                "assignment RHS has no recognised signal or constant",
                statement.line,
            )
            return
        simple = (
            _SIMPLE_REFERENCE_RE.fullmatch(rhs_text.strip()) is not None
            or _parse_constant_bits(rhs_text.strip()) is not None
            or _matching_outer_braces(rhs_text.strip())
        )
        if simple:
            source_bits = _resize_bits(source_bits, len(destinations))
        elif len(destinations) > 1:
            self.diagnostics.record(
                "conservative_expression_dependencies",
                "multi-bit expression uses conservative dependency edges",
                statement.line,
            )
        self._drive(
            destinations,
            [source_bits],
            node_type,
            is_module,
            statement.line,
            triangular=node_type in ("add", "sub", "mult"),
        )

    def _port_direction(self, cell_type: str, port: str) -> Optional[str]:
        mapping = self.cell_ports.get(cell_type.lower())
        if mapping is not None and port.lower() in mapping:
            return mapping[port.lower()]
        if _is_output_port(cell_type, port):
            return "output"
        return None

    def _parse_instance_statement(self, statement: Statement) -> None:
        try:
            cell_type, instance_name, port_text = _parse_instance(statement.text)
            style, connections = _parse_connections(port_text)
        except CommercialNetlistError as error:
            self._fail_or_record("invalid_instance", str(error), statement.line)
            return
        self._instance_count += 1
        operation, is_module = _cell_operation(cell_type)
        sequential = _SEQUENTIAL_CELL_RE.search(cell_type) is not None

        if style == "positional":
            if cell_type.lower() not in _PRIMITIVES and cell_type.lower() not in self.cell_ports:
                self._fail_or_record(
                    "unknown_positional_cell_interface",
                    "cannot infer positional ports for {} {}".format(cell_type, instance_name),
                    statement.line,
                )
                return
            output_connections = connections[:1]
            input_connections = connections[1:]
        else:
            output_connections = []
            input_connections = []
            for port, expression in connections:
                direction = self._port_direction(cell_type, port)
                if direction in ("output", "inout"):
                    output_connections.append((port, expression))
                if direction in ("input", "inout") or direction is None:
                    input_connections.append((port, expression))
            if not output_connections:
                self._fail_or_record(
                    "unknown_output_port",
                    "cannot identify an output port for {} {} (ports: {})".format(
                        cell_type, instance_name, ", ".join(port for port, _ in connections)
                    ),
                    statement.line,
                )
                return

        input_groups = []
        control_groups = []
        data_groups = []
        for port, expression in input_connections:
            if not expression:
                continue
            bits = expand_expression_bits(
                expression, self.signals, self.diagnostics, statement.line
            )
            input_groups.append(bits)
            lower_port = port.lower()
            if lower_port.startswith(("sel", "s", "en", "ce", "re", "we", "clk", "rst")):
                control_groups.append(bits)
            else:
                data_groups.append(bits)

        for _, expression in output_connections:
            destinations = expand_expression_bits(
                expression, self.signals, self.diagnostics, statement.line
            )
            if not destinations:
                continue
            if sequential:
                if self.graph is not None:
                    for destination_name in destinations:
                        destination = self._ensure_graph_node(destination_name)
                        self.graph.node_types[destination] = "input"
                        self.graph.is_input[destination] = True
                self.diagnostics.record(
                    "sequential_boundary",
                    "treated {} {} output as a timing boundary".format(
                        cell_type, instance_name
                    ),
                    statement.line,
                )
                continue

            if operation == "mux" and data_groups:
                groups = list(data_groups) + list(control_groups)
            else:
                groups = input_groups
            self._drive(
                destinations,
                groups,
                operation,
                is_module,
                statement.line,
                triangular=operation in ("add", "sub", "mult"),
            )

    def parse(self) -> CommercialParseResult:
        if not self.netlist.is_file():
            raise FileNotFoundError(self.netlist)

        for statement in iter_verilog_statements(self.netlist):
            text = _ATTRIBUTE_RE.sub("", statement.text).strip()
            module_name = _parse_module_name(text)
            if module_name is not None:
                selected = self.top_module is None or module_name == self.top_module
                self._inside_selected = selected and not self._seen_selected
                if self._inside_selected:
                    self._selected_module = module_name
                    self._seen_selected = True
                    _parse_ansi_module_ports(
                        text, statement.line, self.signals, self.diagnostics
                    )
                continue

            if text == "endmodule":
                if self._inside_selected:
                    self._inside_selected = False
                    break
                continue
            if not self._inside_selected:
                continue

            self._statement_count += 1
            if _parse_declaration(
                text, statement.line, self.signals, self.diagnostics
            ):
                continue
            lower = text.lower().lstrip()
            if lower.startswith("assign"):
                self._parse_assign(Statement(text, statement.line))
                continue
            if lower.startswith(_IGNORED_STATEMENT_PREFIXES):
                continue
            if _PROCEDURAL_RE.match(lower):
                self._fail_or_record(
                    "procedural_top_level_logic",
                    "procedural logic is unsupported in selected structural module",
                    statement.line,
                )
                continue
            self._parse_instance_statement(Statement(text, statement.line))

        if self._selected_module is None:
            requested = self.top_module or "first module"
            raise CommercialNetlistError(
                "could not find {} in {}".format(requested, self.netlist)
            )
        self._materialise_declared_signals()
        return CommercialParseResult(
            netlist=self.netlist,
            module_name=self._selected_module,
            mode=self.mode,
            signals=self.signals,
            graph=self.graph,
            diagnostics=self.diagnostics,
            statement_count=self._statement_count,
            assignment_count=self._assignment_count,
            instance_count=self._instance_count,
        )
