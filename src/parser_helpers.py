from collections import deque
import re


_DECLARATION_QUALIFIERS = {'wire', 'reg', 'logic', 'signed', 'unsigned'}
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', flags=re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'//[^\r\n]*')
_DECLARATION_RE = re.compile(r'^(input|output|wire)\b(.*)$', flags=re.DOTALL)
_DECLARATION_RANGE_RE = re.compile(
    r'^\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]\s*(.*)$'
)
_NAMED_PORT_RE = re.compile(r'\.([A-Za-z_][A-Za-z0-9_$]*)\s*\(')
_CASE_PARSER_CONTEXT = None
_LABEL_MARKER = 'pin to pin level synthesised'


def filter_list(values, indices):
    return [values[index] for index in indices]


def inclusive_range(left, right):
    """Return an inclusive Verilog range in its declared direction."""
    step = 1 if right >= left else -1
    return list(range(left, right + step, step))


def split_top_level(text, delimiter=','):
    """Split on a delimiter while respecting (), [] and {} nesting."""
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    parts = []
    start = 0
    for index, char in enumerate(text):
        if char in '([{':
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                raise ValueError('unbalanced expression: {!r}'.format(text))
        elif char == delimiter and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if stack:
        raise ValueError('unbalanced expression: {!r}'.format(text))
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def split_verilog_statements(content):
    """Split the structural netlist after removing Verilog comments."""
    content = _BLOCK_COMMENT_RE.sub(' ', content)
    content = _LINE_COMMENT_RE.sub(' ', content)
    return [statement.strip() for statement in content.split(';') if statement.strip()]


def recover_structural_instance(statement):
    """Recover the last complete instance header from a malformed statement.

    Some legacy TD netlists contain a truncated named-port fragment followed by
    another instance in the same semicolon-delimited statement. The original
    parser discarded the prefix and parsed the final instance. Return that
    recovered suffix, or ``None`` when the statement has no valid instance.
    """
    open_index = statement.find('(')
    if open_index >= 0 and len(statement[:open_index].strip().split()) == 2:
        return statement

    header_open = statement.rfind(' (')
    if header_open < 0:
        return None
    prefix = statement[:header_open]
    header = prefix[prefix.rfind(',') + 1:].strip()
    if len(header.split()) != 2:
        return None
    return header + statement[header_open:]


def parse_declaration(statement):
    """Parse a structural input/output/wire declaration."""
    statement = statement.lstrip()
    if not statement.startswith(('input', 'output', 'wire')):
        return None
    match = _DECLARATION_RE.match(statement)
    if match is None:
        return None

    kind, remainder = match.groups()
    tokens = remainder.strip().split()
    while tokens and tokens[0] in _DECLARATION_QUALIFIERS:
        tokens.pop(0)
    remainder = ' '.join(tokens)

    bit_range = None
    range_match = _DECLARATION_RANGE_RE.match(remainder)
    if range_match is not None:
        left, right, remainder = range_match.groups()
        bit_range = (int(left), int(right))

    names = split_top_level(remainder)
    if not names:
        raise ValueError('declaration has no signal name: {!r}'.format(statement))
    return kind, bit_range, names


def _matching_parenthesis_end(text, open_index):
    depth = 0
    quote = None
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char == '"':
            quote = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return index
    raise ValueError('unterminated parenthesized expression: {!r}'.format(text))


def _structural_instance_parts(statement):
    text = statement.strip()
    cell_match = re.match(r'([^\s#(]+)', text)
    if cell_match is None:
        raise ValueError('instance has no cell type: {!r}'.format(statement))
    cell_type = cell_match.group(1)
    index = cell_match.end()
    while index < len(text) and text[index].isspace():
        index += 1

    if index < len(text) and text[index] == '#':
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != '(':
            raise ValueError('invalid instance parameter block: {!r}'.format(statement))
        index = _matching_parenthesis_end(text, index) + 1
        while index < len(text) and text[index].isspace():
            index += 1

    name_match = re.match(r'([^\s(]+)', text[index:])
    if name_match is None:
        raise ValueError('instance has no name: {!r}'.format(statement))
    instance_name = name_match.group(1)
    index += name_match.end()
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != '(':
        raise ValueError('instance has no port list: {!r}'.format(statement))
    return cell_type, instance_name, index


def parse_instance_header(statement):
    """Return the cell type and instance name, skipping an optional #(...)."""
    cell_type, instance_name, _ = _structural_instance_parts(statement)
    return cell_type, instance_name


def find_instance_port_open(statement):
    """Return the opening parenthesis of the instance port list."""
    _, _, open_index = _structural_instance_parts(statement)
    return open_index


def parse_named_ports(statement):
    """Extract named-port expressions from a structural instance."""
    _, _, open_index = _structural_instance_parts(statement)

    ports = {}
    text = statement[open_index + 1:]
    index = 0
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] == ','):
            index += 1
        if index >= len(text) or text[index] == ')':
            break
        match = _NAMED_PORT_RE.match(text, index)
        if match is None:
            raise ValueError('invalid named-port list near {!r}'.format(text[index:index + 80]))
        port = match.group(1)
        index = match.end()
        depth = 1
        expression_start = index
        while index < len(text) and depth:
            if text[index] == '(':
                depth += 1
            elif text[index] == ')':
                depth -= 1
            index += 1
        if depth:
            raise ValueError('unterminated .{} port in {!r}'.format(port, statement))
        if port in ports:
            raise ValueError('duplicate .{} port in {!r}'.format(port, statement))
        ports[port] = text[expression_start:index - 1].strip()
    return ports


def resize_unsigned_bits(bits, width, zero_factory):
    """Apply unsigned Verilog assignment sizing to an MSB-to-LSB bit list."""
    if width <= 0:
        raise ValueError('target width must be positive')
    bits = list(bits)
    if not bits:
        raise ValueError('cannot resize an empty bit vector')
    if len(bits) < width:
        bits = [zero_factory() for _ in range(width - len(bits))] + bits
    elif len(bits) > width:
        bits = bits[-width:]
    return bits


def build_critical_edges(po_critical_pis, nname2nid, valid_pis=None, valid_pos=None):
    """Build aligned PI/PO/weight arrays and report rejected endpoints."""
    edges = ([], [], [])
    missing_pis = []
    missing_pos = []
    for po, critical_pis in po_critical_pis.items():
        po_nid = nname2nid.get(po)
        if po_nid is None or (valid_pos is not None and po_nid not in valid_pos):
            missing_pos.append(po)
            continue
        for pi, delay in critical_pis:
            pi_nid = nname2nid.get(pi)
            if pi_nid is None or (valid_pis is not None and pi_nid not in valid_pis):
                missing_pis.append(pi)
                continue
            edges[0].append(pi_nid)
            edges[1].append(po_nid)
            edges[2].append(delay)
    validate_pi2po_edges(edges)
    return edges, missing_pis, missing_pos


def filter_pi2po_edges_by_destinations(edges, valid_destinations):
    """Keep only PI-to-PO edges targeting a labeled endpoint."""
    validate_pi2po_edges(edges)
    valid_destinations = set(valid_destinations)
    kept = [
        (source, destination, weight)
        for source, destination, weight in zip(*edges)
        if destination in valid_destinations
    ]
    if not kept:
        return [], [], []
    sources, destinations, weights = zip(*kept)
    return list(sources), list(destinations), list(weights)


def validate_pi2po_edges(edges, context='pi2po edges', allow_empty=True):
    if not isinstance(edges, (tuple, list)) or len(edges) != 3:
        raise ValueError('{} must contain source, destination and weight arrays'.format(context))
    lengths = tuple(len(values) for values in edges)
    if len(set(lengths)) != 1:
        raise ValueError('{} have mismatched lengths {}'.format(context, lengths))
    if not allow_empty and lengths[0] == 0:
        raise ValueError('{} are empty'.format(context))
    return lengths[0]


def node_constant_value(node, nodes):
    """Return 0/1 for parser constants and -2 for a non-constant node."""
    if "1'b0" in node or nodes.get(node, {}).get('ntype') == "1'b0":
        return 0
    if "1'b1" in node or nodes.get(node, {}).get('ntype') == "1'b1":
        return 1
    return -2


def simplify_constants_scan(fo2fi, nodes, gate_functions):
    """Reference constant propagation using repeated full-graph scans."""
    while True:
        num_simplified = 0
        simplified = {}
        for fanout, (fanout_type, cell_name, fanins) in fo2fi.items():
            if fanout_type != 'gate':
                simplified[fanout] = (fanout_type, cell_name, fanins)
                continue

            gate_type = nodes[fanout]['ntype']
            constant_flags = [node_constant_value(node, nodes) for node in fanins]
            if 0 not in constant_flags and 1 not in constant_flags:
                simplified[fanout] = (fanout_type, cell_name, fanins)
                continue

            num_simplified += 1
            output_flag = gate_functions[gate_type](constant_flags)
            gate_type = {
                0: "1'b0",
                1: "1'b1",
                2: 'buf',
                3: 'not',
            }[output_flag]
            nodes[fanout]['ntype'] = gate_type
            if gate_type in ("1'b0", "1'b1"):
                continue

            remaining_fanins = [
                node for node, flag in zip(fanins, constant_flags) if flag == -2
            ]
            assert gate_type not in ('buf', 'not') or remaining_fanins, (
                'fanout {} with no fanin'.format(fanout)
            )
            if remaining_fanins:
                simplified[fanout] = (
                    fanout_type, cell_name, remaining_fanins
                )
        fo2fi = simplified
        if num_simplified == 0:
            return fo2fi


def simplify_constants_worklist(fo2fi, nodes, gate_functions):
    """Propagate constants by revisiting only consumers of changed nodes."""
    simplified = dict(fo2fi)
    consumers = {}
    for fanout, (_, _, fanins) in fo2fi.items():
        for fanin in fanins:
            consumers.setdefault(fanin, []).append(fanout)

    pending = deque()
    queued = set()

    def enqueue(fanout):
        if fanout in simplified and fanout not in queued:
            pending.append(fanout)
            queued.add(fanout)

    for fanout, (fanout_type, _, fanins) in fo2fi.items():
        if fanout_type == 'gate' and any(
                node_constant_value(node, nodes) != -2 for node in fanins):
            enqueue(fanout)

    while pending:
        fanout = pending.popleft()
        queued.remove(fanout)
        record = simplified.get(fanout)
        if record is None:
            continue
        fanout_type, cell_name, fanins = record
        if fanout_type != 'gate':
            continue

        constant_flags = [node_constant_value(node, nodes) for node in fanins]
        if 0 not in constant_flags and 1 not in constant_flags:
            continue

        gate_type = nodes[fanout]['ntype']
        output_flag = gate_functions[gate_type](constant_flags)
        gate_type = {
            0: "1'b0",
            1: "1'b1",
            2: 'buf',
            3: 'not',
        }[output_flag]
        nodes[fanout]['ntype'] = gate_type
        if gate_type in ("1'b0", "1'b1"):
            del simplified[fanout]
            for consumer in consumers.get(fanout, ()):
                enqueue(consumer)
            continue

        remaining_fanins = [
            node for node, flag in zip(fanins, constant_flags) if flag == -2
        ]
        assert gate_type not in ('buf', 'not') or remaining_fanins, (
            'fanout {} with no fanin'.format(fanout)
        )
        if remaining_fanins:
            simplified[fanout] = (fanout_type, cell_name, remaining_fanins)
        else:
            del simplified[fanout]

    return simplified


def initialize_case_parser(context):
    """Install immutable design metadata in a case-parser worker."""
    global _CASE_PARSER_CONTEXT
    _CASE_PARSER_CONTEXT = context


def parse_case_task(task):
    """Parse one case into model-ready arrays using worker-local metadata."""
    if _CASE_PARSER_CONTEXT is None:
        raise RuntimeError('case parser context has not been initialized')

    case_index, file_path = task
    context = _CASE_PARSER_CONTEXT
    pi_delay, po_labels, po_paths = parse_golden_file(file_path, keep_all_paths=True)

    pi_names = context['pi_names']
    if any(name not in pi_delay for name in pi_names):
        return case_index, None, None, None, None, 0, 0

    pi_delays = [pi_delay[name] for name in pi_names]
    po_values = [po_labels.get(name, -1) for name in context['po_names']]
    po_residuals = [
        (
            po_labels[name] - context['base_po_labels'][name]
            if po_labels.get(name, -1) >= 0
            and context['base_po_labels'].get(name, -1) >= 0
            else -1
        )
        for name in context['po_names']
    ]

    edges = None
    dropped_pis = 0
    dropped_pos = 0
    if po_paths:
        edges, missing_pis, missing_pos = build_critical_edges(
            po_paths,
            context['endpoint_name_to_nid'],
            valid_pis=context['valid_pis'],
            valid_pos=context['valid_pos'],
        )
        dropped_pis = len(missing_pis)
        dropped_pos = len(missing_pos)

    if len(pi_delays) != len(context['valid_pis']) or len(po_values) != len(
            context['valid_pos']):
        raise ValueError(
            '{} case {} feature/endpoint count mismatch'.format(
                context['design_name'], case_index
            )
        )
    return (
        case_index,
        pi_delays,
        po_values,
        po_residuals,
        edges,
        dropped_pis,
        dropped_pos,
    )


def parse_golden_endpoint_names(file_path):
    """Return endpoints that have at least one label in a timing-label file."""
    endpoint_names = set()
    marker_found = False
    with open(file_path, 'r', encoding='utf-8', errors='strict') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not marker_found:
                if stripped.lstrip('/').strip() == _LABEL_MARKER:
                    marker_found = True
                continue

            words = stripped.split()
            if not words or words[0].startswith('//'):
                continue
            if len(words) not in (2, 3):
                raise ValueError(
                    'wrong PO info at {}:{}: {!r}'.format(
                        file_path, line_number, line.rstrip('\r\n')
                    )
                )
            if len(words) == 3 and 'clk' in words[1]:
                continue
            endpoint_names.add(words[0].rstrip(','))

    if not marker_found:
        raise ValueError('no PO info in {}'.format(file_path))
    return endpoint_names


def parse_golden_file(file_path, keep_all_paths):
    """Parse one timing-label file with whitespace/newline normalization."""
    with open(file_path, 'r', encoding='utf-8', errors='strict') as handle:
        lines = handle.read().splitlines()

    marker_index = next(
        (index for index, line in enumerate(lines)
         if line.strip().lstrip('/').strip() == _LABEL_MARKER),
        None,
    )
    if marker_index is None:
        raise ValueError('no PO info in {}'.format(file_path))

    pi_delay = {}
    for line_number, line in enumerate(lines[:marker_index], start=1):
        words = line.split()
        if not words or words[0].startswith('//'):
            continue
        if len(words) != 2:
            raise ValueError('wrong PI info at {}:{}: {!r}'.format(file_path, line_number, line))
        pi, delay = words
        pi_delay[pi.rstrip(',')] = int(delay)

    po_labels = {}
    po_paths = {}
    for line_number, line in enumerate(lines[marker_index + 1:], start=marker_index + 2):
        words = line.split()
        if not words or words[0].startswith('//'):
            continue
        if len(words) == 2:
            po, delay = words
            po = po.rstrip(',')
            delay = int(delay)
            current = po_labels.get(po, 0)
            po_labels[po] = delay if delay > current else current
            continue
        if len(words) != 3:
            raise ValueError('wrong PO info at {}:{}: {!r}'.format(file_path, line_number, line))
        po, pi, delay = words
        po = po.rstrip(',')
        pi = pi.rstrip(',')
        if 'clk' in pi:
            continue
        delay = int(delay)
        current = po_labels.get(po, 0)
        po_labels[po] = delay if delay > current else current
        po_paths.setdefault(po, []).append((pi, delay))

    if keep_all_paths:
        return pi_delay, po_labels, po_paths

    po_critical_pis = {}
    for po, paths in po_paths.items():
        max_delay = po_labels[po]
        po_critical_pis[po] = [(pi, 1) for pi, delay in paths if delay == max_delay]
    return pi_delay, po_labels, po_critical_pis
