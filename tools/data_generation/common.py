"""Shared helpers for the NUA-Timer raw-data generation pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


LABEL_MARKER = "// pin to pin level synthesised"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_td_netlist_content(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a generated netlist while ignoring TD's volatile timestamp."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        first_line = handle.readline()
        second_line = handle.readline()
        digest.update(first_line)
        if not (
            first_line.startswith(b"// Verilog netlist created by Tang Dynasty")
            and second_line.startswith(b"// ")
        ):
            digest.update(second_line)
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_case_name(case_name: str) -> Tuple[str, int]:
    try:
        design, index_text = case_name.rsplit("_", 1)
        return design, int(index_text)
    except (ValueError, AttributeError):
        raise ValueError("invalid case directory name: {}".format(case_name))


def iter_case_dirs(
    projects_dir: Path,
    designs: Optional[Set[str]] = None,
    indices: Optional[Set[int]] = None,
) -> List[Path]:
    cases: List[Tuple[str, int, Path]] = []
    for path in projects_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            design, index = split_case_name(path.name)
        except ValueError:
            continue
        if designs is not None and design not in designs:
            continue
        if indices is not None and index not in indices:
            continue
        cases.append((design, index, path))
    return [item[2] for item in sorted(cases, key=lambda item: (item[0], item[1]))]


def parse_index_spec(spec: Optional[str]) -> Optional[Set[int]]:
    if not spec:
        return None
    result: Set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError("invalid descending case range: {}".format(token))
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    return result


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
_INPUT_DECL_RE = re.compile(r"\binput\b(?P<body>.*?);", re.DOTALL)
_RANGE_RE = re.compile(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def parse_input_ranges(netlist: Path) -> Dict[str, Tuple[int, int]]:
    """Return packed input ranges from a generated, elaborated Verilog netlist."""

    text = netlist.read_text(encoding="utf-8", errors="ignore")
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    ranges: Dict[str, Tuple[int, int]] = {}

    for declaration in _INPUT_DECL_RE.finditer(text):
        body = declaration.group("body")
        packed = _RANGE_RE.search(body)
        if packed is None:
            continue
        msb, lsb = int(packed.group(1)), int(packed.group(2))
        body = _RANGE_RE.sub(" ", body, count=1)
        body = re.sub(r"\b(?:wire|reg|logic|signed|unsigned|tri|var)\b", " ", body)
        for item in body.split(","):
            identifiers = _IDENTIFIER_RE.findall(item)
            if identifiers:
                ranges[identifiers[-1]] = (msb, lsb)
    return ranges


def read_input_levels(golden_file: Path) -> List[Tuple[str, int]]:
    inputs: List[Tuple[str, int]] = []
    for raw_line in golden_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line == LABEL_MARKER:
            break
        if not line or line.startswith("//"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("invalid PI label line in {}: {}".format(golden_file, line))
        inputs.append((fields[0], int(fields[1])))
    if not inputs:
        raise ValueError("no PI labels found in {}".format(golden_file))
    return inputs


def normalize_input_levels(
    inputs: Sequence[Tuple[str, int]],
    input_ranges: Mapping[str, Tuple[int, int]],
) -> List[Tuple[str, int]]:
    normalized: List[Tuple[str, int]] = []
    seen: Set[str] = set()
    for pin, level in inputs:
        if "[" in pin or pin not in input_ranges:
            expanded = [(pin, level)]
        else:
            msb, lsb = input_ranges[pin]
            low, high = min(msb, lsb), max(msb, lsb)
            expanded = [("{}[{}]".format(pin, bit), level) for bit in range(low, high + 1)]
        for expanded_pin, expanded_level in expanded:
            if expanded_pin in seen:
                continue
            seen.add(expanded_pin)
            normalized.append((expanded_pin, expanded_level))
    return normalized


_BEGIN_RE = re.compile(r"^Begin\s+Point\s*:\s*(\S+)", re.MULTILINE)
_END_RE = re.compile(r"^End\s+Point\s*:\s*(\S+)", re.MULTILINE)
_LEVEL_RE = re.compile(r"^Logic\s+Level\s*:\s*(.*)$", re.MULTILINE)
_REGISTER_PIN_RE = re.compile(r"^(?P<register>.+)_(?P<port>[qd])(?P<bits>(?:\[\d+\])*)$")


def _adder_logic_level(chunk: str) -> int:
    sequence: List[str] = []
    for line in chunk.splitlines():
        cell_type = line.strip().split(None, 1)[0] if line.strip() else ""
        if cell_type in ("ADDER", "LUT"):
            if cell_type == "ADDER" and sequence and sequence[-1] == "ADDER":
                continue
            sequence.append(cell_type)
    return sum(2 if cell_type == "ADDER" else 1 for cell_type in sequence)


def _is_same_register_q_to_d(begin: str, end: str) -> bool:
    begin_match = _REGISTER_PIN_RE.fullmatch(begin)
    end_match = _REGISTER_PIN_RE.fullmatch(end)
    if begin_match is None or end_match is None:
        return False
    return (
        "reg" in begin_match.group("register")
        and begin_match.group("port") == "q"
        and end_match.group("port") == "d"
        and begin_match.group("register") == end_match.group("register")
        and begin_match.group("bits") == end_match.group("bits")
    )


def _parse_timing_chunk(chunk: str) -> Optional[Tuple[str, str, int]]:
    begin_match = _BEGIN_RE.search(chunk)
    end_match = _END_RE.search(chunk)
    level_match = _LEVEL_RE.search(chunk)
    if begin_match is None or end_match is None or level_match is None:
        return None
    level_text = level_match.group(1)
    if "ADDER" in level_text:
        level = _adder_logic_level(chunk)
    else:
        number = re.search(r"-?\d+", level_text)
        if number is None:
            return None
        level = int(number.group(0))
    begin, end = begin_match.group(1), end_match.group(1)
    if _is_same_register_q_to_d(begin, end):
        return None
    return begin, end, level


def iter_timing_paths(report_file: Path) -> Iterator[Tuple[str, str, int]]:
    """Yield timing paths without loading a potentially multi-GB report."""

    chunk: List[str] = []
    found = False
    with report_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if re.match(r"^Slack\b", line):
                if chunk:
                    path = _parse_timing_chunk("".join(chunk))
                    if path is not None:
                        found = True
                        yield path
                chunk = [line]
            elif chunk:
                chunk.append(line)
    if chunk:
        path = _parse_timing_chunk("".join(chunk))
        if path is not None:
            found = True
            yield path
    if not found:
        raise ValueError("no timing paths found in {}".format(report_file))


def parse_timing_paths(report_file: Path) -> List[Tuple[str, str, int]]:
    return list(iter_timing_paths(report_file))


def build_endpoint_labels(
    inputs: Sequence[Tuple[str, int]],
    paths: Iterable[Tuple[str, str, int]],
    diagnostics: Optional[MutableMapping[str, int]] = None,
) -> List[Tuple[str, str, int]]:
    """Reduce NUIAT-start timing paths to endpoint labels."""

    input_levels = dict(inputs)
    best: Dict[str, int] = {}
    critical: Dict[str, Set[Tuple[str, int]]] = {}
    timing_path_count = 0
    nuiat_startpoint_path_count = 0
    nuiat_startpoints: Set[str] = set()
    unexpected_startpoint_path_count = 0
    unexpected_startpoints: Set[str] = set()
    for begin, end, logic_level in paths:
        timing_path_count += 1
        if begin not in input_levels:
            unexpected_startpoint_path_count += 1
            unexpected_startpoints.add(begin)
            continue
        nuiat_startpoint_path_count += 1
        nuiat_startpoints.add(begin)
        total_level = input_levels[begin] + logic_level
        if end not in best or total_level > best[end]:
            best[end] = total_level
            critical[end] = {(begin, total_level)}
        elif total_level == best[end]:
            critical[end].add((begin, total_level))
    if diagnostics is not None:
        diagnostics.update(
            {
                "timing_path_count": timing_path_count,
                "nuiat_startpoint_path_count": nuiat_startpoint_path_count,
                "nuiat_startpoint_count": len(nuiat_startpoints),
                "unexpected_startpoint_path_count": unexpected_startpoint_path_count,
                "unexpected_startpoint_count": len(unexpected_startpoints),
            }
        )
    if unexpected_startpoints:
        sample = ", ".join(sorted(unexpected_startpoints)[:5])
        raise ValueError(
            "timing report contains {} paths from {} startpoints outside the "
            "current case NUIAT table; examples: {}".format(
                unexpected_startpoint_path_count,
                len(unexpected_startpoints),
                sample,
            )
        )
    if not critical:
        raise ValueError("timing report has no NUIAT-start paths")
    return [
        (endpoint, begin, total)
        for endpoint in sorted(critical)
        for begin, total in sorted(critical[endpoint])
    ]


def write_labeled_golden(
    output_file: Path,
    inputs: Sequence[Tuple[str, int]],
    labels: Sequence[Tuple[str, str, int]],
) -> None:
    lines = ["// input constrains", "// pin, level"]
    lines.extend("{} {}".format(pin, level) for pin, level in inputs)
    lines.extend([LABEL_MARKER, "// outpin, critical input, max level"])
    lines.extend("{}, {}, {}".format(endpoint, begin, total) for endpoint, begin, total in labels)
    atomic_write_text(output_file, "\n".join(lines) + "\n")


def validate_labeled_golden(
    path: Path,
    require_nuiat_startpoints: bool = False,
) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if LABEL_MARKER not in text:
        raise ValueError("missing label marker in {}".format(path))
    before, after = text.split(LABEL_MARKER, 1)
    pi_names = {
        line.split()[0]
        for line in before.splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    }
    endpoints: Set[str] = set()
    critical_inputs: Set[str] = set()
    label_count = 0
    for raw_line in after.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        fields = line.replace(",", "").split()
        if len(fields) not in (2, 3):
            raise ValueError("invalid endpoint label in {}: {}".format(path, line))
        if require_nuiat_startpoints:
            if len(fields) != 3:
                raise ValueError(
                    "endpoint label has no critical input in {}: {}".format(path, line)
                )
            if fields[1] not in pi_names:
                raise ValueError(
                    "critical input {} is not a NUIAT PI in {}".format(
                        fields[1], path
                    )
                )
        if len(fields) == 3:
            critical_inputs.add(fields[1])
        int(fields[-1])
        endpoints.add(fields[0])
        label_count += 1
    if not endpoints:
        raise ValueError("no endpoint labels in {}".format(path))
    return {
        "pi_count": len(pi_names),
        "endpoint_count": len(endpoints),
        "label_count": label_count,
        "critical_input_count": len(critical_inputs),
        "endpoints": sorted(endpoints),
    }


def _tcl_path(path: Path) -> str:
    text = str(path.resolve())
    if "{" in text or "}" in text:
        raise ValueError("Tcl path contains an unsupported brace: {}".format(path))
    return "{{{}}}".format(text)


def _validate_tcl_list_items(items: Sequence[str], description: str) -> List[str]:
    result = list(items)
    if not result:
        raise ValueError("{} must not be empty".format(description))
    if len(set(result)) != len(result):
        raise ValueError("{} contains duplicate entries".format(description))
    for item in result:
        if not item or re.search(r"[\s{}]", item):
            raise ValueError(
                "{} contains a Tcl-unsafe item: {!r}".format(description, item)
            )
    return result


def _restricted_timing_report_tcl(
    indent: str,
    startpoints: Sequence[str],
    path_num: int,
    endpoint_num: int,
) -> List[str]:
    names = _validate_tcl_list_items(startpoints, "NUIAT startpoints")
    lines = ["{}set nua_startpoint_names {{".format(indent)]
    lines.extend("{}    {}".format(indent, name) for name in names)
    lines.extend(
        [
            "{}}}".format(indent),
            "{}set nua_startpoints "
            "[get_ports -nowarn $nua_startpoint_names]".format(indent),
            "{}if {{[llength $nua_startpoints] != {}}} {{".format(
                indent, len(names)
            ),
            '{}    puts stderr "NUATIMER_ERROR: expected {} NUIAT ports, '
            'resolved [llength $nua_startpoints]"'.format(indent, len(names)),
            "{}    exit 2".format(indent),
            "{}}}".format(indent),
            "{}report_timing_path -from $nua_startpoints "
            "-path_num {} -ep_num {} -file timing.rpt".format(
                indent, path_num, endpoint_num
            ),
        ]
    )
    return lines


def rewrite_sta_tcl(
    text: str,
    path_num: int,
    endpoint_num: int,
    startpoints: Sequence[str],
    checkpoint: Optional[Path] = None,
    netlist: Optional[Path] = None,
    keep_debug_netlists: bool = False,
    qor_monitor: bool = False,
    td_threads: str = "auto",
) -> str:
    if checkpoint is not None and netlist is not None:
        raise ValueError("checkpoint and netlist are mutually exclusive")

    lines = []
    found_report = False
    found_design_load = False
    for line in text.splitlines():
        if re.match(r"\s*set_param\s+flow\s+(?:qor_monitor|thread)\b", line):
            continue
        if re.match(r"\s*read_verilog\b", line):
            found_design_load = True
            indent = line[:len(line) - len(line.lstrip())]
            if checkpoint is not None:
                line = "{}import_db {}".format(indent, _tcl_path(checkpoint))
            elif netlist is not None:
                line = re.sub(r"-file\s+\S+", "-file {}".format(_tcl_path(netlist)), line)
            lines.append(line)
            lines.append(
                "{}set_param flow qor_monitor {}".format(
                    indent, "on" if qor_monitor else "off"
                )
            )
            lines.append("{}set_param flow thread {}".format(indent, td_threads))
            continue
        if (
            not keep_debug_netlists
            and re.match(r"\s*write_verilog\s+(?:read|gate)\.v\b", line)
        ):
            continue
        if re.match(r"\s*report_timing_(?:summary|path)\b", line):
            found_report = True
            indent = line[:len(line) - len(line.lstrip())]
            lines.extend(
                _restricted_timing_report_tcl(
                    indent,
                    startpoints,
                    path_num,
                    endpoint_num,
                )
            )
            continue
        lines.append(line)
    if not found_design_load:
        raise ValueError("run.tcl does not contain read_verilog")
    if not found_report:
        raise ValueError("run.tcl does not contain a timing report command")
    return "\n".join(lines) + "\n"


def build_read_checkpoint_tcl(
    text: str,
    netlist: Path,
    checkpoint: Path,
    td_threads: str = "auto",
) -> str:
    """Build a minimal read/elaborate flow that exports a reusable checkpoint."""

    lines = []
    proc_name = None
    found_read = False
    for line in text.splitlines():
        if proc_name is None:
            proc_match = re.match(r"\s*proc\s+([^\s{]+)", line)
            if proc_match is not None:
                proc_name = proc_match.group(1)
        if re.match(r"\s*set_param\s+flow\s+thread\b", line):
            continue
        if re.match(r"\s*read_verilog\b", line):
            found_read = True
            indent = line[:len(line) - len(line.lstrip())]
            line = re.sub(r"-file\s+\S+", "-file {}".format(_tcl_path(netlist)), line)
            lines.append(line)
            lines.append("{}set_param flow thread {}".format(indent, td_threads))
            lines.append("{}export_db {}".format(indent, _tcl_path(checkpoint)))
            lines.append("{}exit".format(indent))
            lines.append("}")
            break
        lines.append(line)
    if proc_name is None or not found_read:
        raise ValueError("run.tcl does not contain a readable Tcl procedure")
    lines.append("set state(1_td) [{}]".format(proc_name))
    return "\n".join(lines) + "\n"


def extract_first_module(source: Path, destination: Path) -> None:
    """Copy the first generated module while removing synthesis-only regions."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(destination.parent), delete=False
    ) as output:
        temporary = Path(output.name)
        in_translate_off = False
        saw_module = False
        with source.open("r", encoding="utf-8", errors="ignore") as input_handle:
            for line in input_handle:
                if "synthesis translate_off" in line:
                    in_translate_off = True
                    continue
                if "synthesis translate_on" in line:
                    in_translate_off = False
                    continue
                if in_translate_off:
                    continue
                if re.match(r"\s*module\b", line):
                    saw_module = True
                if saw_module:
                    output.write(line)
                if saw_module and re.match(r"\s*endmodule\b", line):
                    break
    if not saw_module:
        temporary.unlink(missing_ok=True)
        raise ValueError("no module found in {}".format(source))
    os.replace(str(temporary), str(destination))


_READ_PORT_MODULE_RE = re.compile(
    r"^(?P<prefix>\s*module\s+read_port_[A-Za-z0-9_$]+)(?P<suffix>.*)$"
)


def prepare_sta_netlist(source: Path, destination: Path) -> Dict[str, int]:
    """Prepare a generated TD netlist to be read back by the STA flow."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    patched_modules = 0
    removed_lines = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(destination.parent), delete=False
    ) as output:
        temporary = Path(output.name)
        in_translate_off = False
        with source.open("r", encoding="utf-8", errors="ignore") as input_handle:
            for line in input_handle:
                if "synthesis translate_off" in line:
                    in_translate_off = True
                    removed_lines += 1
                    continue
                if "synthesis translate_on" in line:
                    in_translate_off = False
                    removed_lines += 1
                    continue
                if in_translate_off:
                    removed_lines += 1
                    continue

                match = _READ_PORT_MODULE_RE.match(line)
                if match is not None:
                    suffix = match.group("suffix")
                    if "#" in suffix:
                        temporary.unlink(missing_ok=True)
                        raise ValueError(
                            "read_port module already has a parameter list: {}".format(
                                line.strip()
                            )
                        )
                    line = (
                        match.group("prefix")
                        + " #(parameter INIT_VALUE = 0, "
                        + 'parameter RAM_INIT_STATE = "0")'
                        + suffix
                        + "\n"
                    )
                    patched_modules += 1
                output.write(line)
    os.replace(str(temporary), str(destination))
    return {
        "read_port_modules_patched": patched_modules,
        "translate_off_lines_removed": removed_lines,
    }
