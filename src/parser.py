import copy
import torch as th
import dgl
import multiprocessing as mp
import os
import pickle
import random
import re
from options import get_options
import tee
from parser_graph_utils import (
    edge_source_level_ratings,
    gen_topo,
    get_pi2po_edges,
    graph_filter,
)
from parser_helpers import (
    filter_list,
    find_instance_port_open,
    inclusive_range,
    initialize_case_parser,
    parse_case_task,
    parse_declaration,
    parse_golden_endpoint_names,
    parse_golden_file,
    parse_instance_header,
    parse_named_ports,
    recover_structural_instance,
    resize_unsigned_bits,
    simplify_constants_scan,
    simplify_constants_worklist,
    split_top_level,
    split_verilog_statements,
    filter_pi2po_edges_by_destinations,
    validate_pi2po_edges,
)
from time import time


_ESCAPED_BIT_SELECT_RE = re.compile(r'(?<=\S)\s+\[(\d+)\]')
_ASSIGN_RE = re.compile(r'^assign\s+(.+?)\s*=\s*(.+)$', flags=re.DOTALL)
_WHITESPACE_RE = re.compile(r'\s+')
_CONSTANT_RE = re.compile(r"(\d+)?'[bB]([01xXzZ_]+)")
_PART_SELECT_RE = re.compile(r'(.+?)\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]')
_REGISTER_VECTOR_SUFFIX_RE = re.compile(r'(?:\[\d+\])+$')

options = get_options() if __name__ == '__main__' else get_options([])
rawdata_path = options.rawdata_path
data_savepath = options.data_savepath
flag_split = options.flag_split
flag_group = options.flag_group
if __name__ == '__main__':
    os.makedirs(data_savepath, exist_ok=True)

ntype_file = os.path.join(
    os.path.dirname(os.path.normpath(data_savepath)), 'ntype2id.pkl'
)

ntype2id = {'input':0, "1'b0":1, "1'b1":2}
ntype2id_gate = {'input':0, "1'b0":1, "1'b1":2}
ntype2id_module = {}

# outputs:
#   0: constant 1'b0
#   1: constant 1'b1
#   2: buf
#   3: not
def func_xor(inputs):
    assert len(inputs) == 2
    sum_i = sum(inputs)
    if sum_i==-2:
        return 2
    elif sum_i==-1:
        return 3
    else:
        return sum_i%2
def func_xnor(inputs):
    assert len(inputs) == 2
    sum_i =sum(inputs)
    if sum_i==-2:
        return 3
    elif sum_i==-1:
        return 2
    else:
        return (sum_i+1)%2

def func_or(inputs):
    assert len(inputs) == 2
    sum_i = sum(inputs)
    if sum_i==-2:
        return 2
    elif sum_i==-1:
        return 1
    else:
        return 1 if sum_i>=1 else 0
def func_and(inputs):
    assert len(inputs) == 2
    sum_i = sum(inputs)
    if sum_i==-2:
        return 0
    elif sum_i==-1:
        return 2
    else:
        return 1 if sum_i==2 else 0

def func_not(inputs):
    assert len(inputs)==1
    return (inputs[0]+1)%2

def func_buf(inputs):
    assert len(inputs)==1
    return inputs[0]

gate_func_map = {
    'xor':func_xor,
    'xnor':func_xnor,
    'and':func_and,
    'or':func_or,
    'not':func_not,
    'buf':func_buf
}


def edges_srcLevel_rating(graph,nodes_level,etype):
    src, dst = graph.edges(etype=etype, form='uv')
    return edge_source_level_ratings(src, dst, th.as_tensor(nodes_level))

class Parser:
    def __init__(
            self, netlist_file, golden_file, flag_log=False,
            po_label_names=None):
        self.design_name = os.path.split(netlist_file)[-1].split('.')[0]
        if '_case' in self.design_name:
            self.design_name = self.design_name[:self.design_name.rfind('_case')]
        self.flag_log = flag_log
        self.flag_runtime = getattr(options, 'log_level', 0) >= 1
        self.netlist_file = netlist_file
        self.golden_file = golden_file
        self.structural_audit = False
        # self.design_path = design_dir
        #print(self.design_name,design_dir)
        self.wires_width = {}
        self.const0_index = 0
        self.const1_index = 0
        self.nodes = {}
        self.edges = [] # edges: List[Tuple[str, str, Dict]] = []  # a list of (src, dst, {key: value})
        self.fo2fi = {}
        self.pi_delay = {}
        self.po_labels = {}
        self.po_label_names = (
            None if po_label_names is None else set(po_label_names)
        )
        self.nicknames = {}
        self.parser_diagnostics = {
            'recovered_malformed_instances': 0,
            'skipped_malformed_instances': 0,
        }
        #self.buf_types = ['buf','not']
        self.buf_types = ['buf']

    def is_constant(self,node):
        if "1'b0" in node or self.nodes.get(node,{}).get('ntype')=="1'b0":
            return 0
        elif "1'b1" in node or self.nodes.get(node,{}).get('ntype')=="1'b1":
            return 1
        else:
            return -2

    def get_moduleIOnodes(self, sentence, output_port='o'):
        io_wires = parse_named_ports(sentence)
        if output_port not in io_wires:
            raise ValueError(
                'named-port instance has no .{} port; ports={}, declaration={!r}'.format(
                    output_port, sorted(io_wires), sentence[:300]
                )
            )
        if output_port != 'o':
            io_wires['o'] = io_wires.pop(output_port)

        io_nodes = {p: self.parse_wire(w) for p, w in io_wires.items()}

        return io_nodes

    def new_constant(self, value):
        if value not in ('0', '1'):
            raise ValueError('unsupported constant bit {!r}'.format(value))
        if value == '0':
            self.const0_index += 1
            index = self.const0_index
        else:
            self.const1_index += 1
            index = self.const1_index
        node_type = "1'b{}".format(value)
        node = '{}_{}'.format(node_type, index) if options.flag_split01 else node_type
        self.nodes[node] = {'ntype': node_type, 'is_po': False, 'is_module': 0}
        return node

    def resize_unsigned_nodes(self, nodes, width):
        return resize_unsigned_bits(nodes, width, lambda: self.new_constant('0'))

    #.i0({3'b000,do_2_b5_n,do_2_b5_n1,do_2_b5_n2,do_2_b5_n3,do_2_b5_n4}),
    # parse the given wire, and return the list of single-bit pins
    def parse_wire(self, wire):
        wire = wire.strip()
        if wire.startswith('{'):
            if not wire.endswith('}'):
                raise ValueError('unterminated concatenation {!r}'.format(wire))
            result = []
            for part in split_top_level(wire[1:-1]):
                result.extend(self.parse_wire(part))
            return result

        constant = _CONSTANT_RE.fullmatch(wire)
        if constant is not None:
            declared_width, raw_values = constant.groups()
            values = raw_values.replace('_', '').lower()
            if declared_width is not None:
                width = int(declared_width)
                values = values[-width:].rjust(width, '0')
            return [self.new_constant('1' if value == '1' else '0') for value in values]

        part_select = _PART_SELECT_RE.fullmatch(wire)
        if part_select is not None:
            wire_name, left, right = part_select.groups()
            return [
                '{}[{}]'.format(wire_name.strip(), bit)
                for bit in inclusive_range(int(left), int(right))
            ]

        if wire in self.wires_width:
            bit_range = self.wires_width[wire]
            if bit_range is None:
                return [wire]
            left, right = bit_range
            return ['{}[{}]'.format(wire, bit) for bit in inclusive_range(left, right)]
        return [wire]

    def register_memory_read_mux(self, io_nodes, gate_name, fo2fi, data_port):
        fanout_nodes = list(reversed(io_nodes['o']))
        data_nodes = list(reversed(io_nodes.get(data_port, [])))
        controls = []
        for port, nodes in io_nodes.items():
            if port not in ('o', data_port):
                controls.extend(nodes)
        output_width = len(fanout_nodes)
        data_width = len(data_nodes)
        if output_width == 0 or data_width == 0:
            raise ValueError('empty memory helper port in {}'.format(gate_name))
        scalar_memory_state = data_width == 1
        if not scalar_memory_state and data_width % output_width != 0:
            raise ValueError(
                'memory helper {} has data width {} not divisible by output width {}'.format(
                    gate_name, data_width, output_width
                )
            )

        for index, fanout_node in enumerate(fanout_nodes):
            if self.nodes.get(fanout_node) is None and 'open' in fanout_node:
                self.nodes[fanout_node] = {
                    'ntype': 'mux', 'is_po': False, 'is_module': 0
                }
            else:
                self.nodes[fanout_node]['ntype'] = 'mux'
                self.nodes[fanout_node]['is_module'] = 0
            data_fanins = data_nodes if scalar_memory_state else data_nodes[index::output_width]
            fanins = data_fanins + controls
            self.nodes[fanout_node]['width'] = len(fanins)
            fo2fi[fanout_node] = fanins
            self.fo2fi[fanout_node] = ('mux', gate_name, fanins)

        ntype2id['mux'] = ntype2id.get('mux', len(ntype2id))
        ntype2id_gate['mux'] = ntype2id_gate.get('mux', len(ntype2id_gate))

    def parse_verilog(self):
        # if 'round7' in self.design_path:
        #     file_path = os.path.join(self.design_path, '{}.v'.format(self.design_name))
        # else:
        #     file_path = os.path.join(self.design_path,'{}_case.v'.format(self.design_name))
        # file_path
        runtime = 0
        if self.flag_runtime:
            start = time()
        with open(self.netlist_file, 'r', encoding='utf-8', errors='strict') as f:
            content = f.read()
        statements = split_verilog_statements(content)
        first_declaration = next(
            (index for index, statement in enumerate(statements)
             if parse_declaration(statement) is not None),
            None,
        )
        if first_declaration is None:
            raise ValueError('netlist has no input/output/wire declarations: {}'.format(
                self.netlist_file
            ))
        statements = statements[first_declaration:]
        if self.flag_runtime:
            runtime += time()-start

        # if 'lt2023' in content and 'lt2024' not in content:
        #     error_region = content[content.find('lt2023'):content.find('lt2025')]
        #     error_region = error_region[:error_region.rfind('.o')]
        #     fixed_region = error_region.replace(';\n',';\n  lt_u1_u1 lt2024 (\n')
        #     #print(fixed_region)
        #     content = content.replace(error_region,fixed_region)
        #     with open(file_path, 'w') as f:
        #         f.write(content)
        #     print('---fix')
            # return None, {}

        buf_o2i, buf_i2o = {}, {}
        fo2fi_bit_position = {}
        assign_i2o = {}
        assign_o2i = {}

        if self.flag_runtime:
            start = time()
        for sentence in statements:
            if not sentence or sentence.strip().startswith('endmodule'):
                continue

            if self.structural_audit and re.search(
                    r'\b(always(?:_ff|_comb|_latch)?|begin|end|case|endcase)\b',
                    sentence,
            ):
                # Procedural RAM/RTL bodies are outside the structural parser
                # grammar. They do not contribute structural graph instances.
                self.parser_diagnostics['skipped_malformed_instances'] += 1
                continue

            sentence = sentence.replace('\\','')
            # Escaped Verilog identifiers end at whitespace, so a following bit
            # select may appear as ``name [3]`` even though declarations expand
            # the same signal as ``name[3]``.
            sentence = _ESCAPED_BIT_SELECT_RE.sub(r'[\1]', sentence)
            declaration = parse_declaration(sentence)
            if declaration is not None:
                wire_type, bit_range, wire_names = declaration
                for wire_name in wire_names:
                    wire_name = wire_name.strip()
                    if '=' in wire_name:
                        raise ValueError('initialized declarations are unsupported: {!r}'.format(
                            sentence
                        ))
                    self.wires_width[wire_name] = bit_range
                    bit_names = (
                        [wire_name]
                        if bit_range is None
                        else ['{}[{}]'.format(wire_name, bit)
                              for bit in inclusive_range(*bit_range)]
                    )
                    for node_name in bit_names:
                        self.nodes[node_name] = {
                            'ntype': wire_type,
                            'is_po': wire_type == 'output',
                            'is_module': 0,
                        }
            elif sentence.strip().startswith('assign'):
                assign_match = _ASSIGN_RE.match(sentence)
                if assign_match is None:
                    raise ValueError('invalid assign statement: {!r}'.format(sentence))
                output_wire, input_wire = assign_match.groups()
                output_nodes = self.parse_wire(output_wire)
                input_nodes = self.resize_unsigned_nodes(
                    self.parse_wire(input_wire), len(output_nodes)
                )
                for output_node, input_node in zip(output_nodes, input_nodes):
                    if output_node not in self.nodes:
                        raise ValueError('assign output {} is undeclared'.format(output_node))
                    self.fo2fi[output_node] = ('gate', 'buf', [input_node])
                    self.nodes[output_node]['ntype'] = 'buf'
            else:
                fo2fi = {}  # {fanout_name:fanin_list}
                sentence = _WHITESPACE_RE.sub(' ', sentence).strip()
                try:
                    gate_parts = parse_instance_header(sentence)
                except ValueError:
                    recovered_sentence = recover_structural_instance(sentence)
                    if recovered_sentence is None:
                        if sentence.lstrip().startswith('.'):
                            self.parser_diagnostics['skipped_malformed_instances'] += 1
                            continue
                        open_index = sentence.find('(')
                        if open_index < 0:
                            if self.structural_audit:
                                self.parser_diagnostics['skipped_malformed_instances'] += 1
                                continue
                            raise ValueError(
                                'invalid structural statement: {!r}'.format(sentence)
                            )
                        if self.structural_audit:
                            self.parser_diagnostics['skipped_malformed_instances'] += 1
                            continue
                        raise ValueError(
                            'invalid gate declaration: {!r}'.format(
                                sentence[:open_index].strip()
                            )
                        )
                    sentence = recovered_sentence
                    gate_parts = parse_instance_header(sentence)
                    self.parser_diagnostics['recovered_malformed_instances'] += 1
                gate_type, gate_name = gate_parts
                open_index = find_instance_port_open(sentence)


                # deal with multiplexer, whose width may be larger than 1
                if 'mux' in gate_type.lower():
                    gate_type = 'mux'
                    try:
                        io_nodes = self.get_moduleIOnodes(sentence)
                    except ValueError:
                        self.parser_diagnostics['skipped_malformed_instances'] += 1
                        continue

                    required_ports = {'i0', 'i1', 'sel', 'o'}
                    if set(io_nodes) != required_ports:
                        self.parser_diagnostics['skipped_malformed_instances'] += 1
                        continue

                    # get the output nodes, and set their gate type;
                    fanout_nodes = io_nodes['o']
                    if not fanout_nodes:
                        raise ValueError('mux {} has no output bits'.format(gate_name))
                    for n in fanout_nodes:
                        if self.nodes.get(n,None) is None and 'open' in n:
                            self.nodes[n] = {'ntype':gate_type,'is_po':False,'is_module':0}
                        else:
                            self.nodes[n]['ntype'] = gate_type
                        ntype2id['mux'] = ntype2id.get('mux',len(ntype2id))
                        ntype2id_gate['mux'] = ntype2id_gate.get('mux', len(ntype2id_gate))
                        fo2fi[n] = []
                        self.fo2fi[n] = ('mux',gate_name,[])
                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        if port == 'o':
                            continue
                        # for port i0,i1: link fi_i[j] with fo[j]
                        elif port.startswith('i'):
                            fanin_nodes = self.resize_unsigned_nodes(
                                fanin_nodes, len(fanout_nodes)
                            )
                            for i,fanout_node in enumerate(fanout_nodes):
                                fanin_node = fanin_nodes[i]
                                fo2fi[fanout_node].append(fanin_node)
                                self.fo2fi[fanout_node][2].append(fanin_node)
                        # for port sel: link all fi_s with fo[j]
                        elif port=='sel':
                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes)
                                self.fo2fi[fanout_node][2].extend(fanin_nodes)
                        else:
                            raise ValueError(
                                'unsupported mux port {} in {}'.format(port, gate_name)
                            )
                # TD emits packed memories through read_port helpers. Model each
                # read bit as a mux over the corresponding packed RAM bits.
                elif gate_type.startswith('read_port_'):
                    io_nodes = self.get_moduleIOnodes(sentence, output_port='rd')
                    self.register_memory_read_mux(
                        io_nodes, gate_name, fo2fi, data_port='ram'
                    )
                elif gate_type.startswith('clock_write_port_'):
                    io_nodes = self.get_moduleIOnodes(sentence, output_port='ram')
                    # A clocked write produces the next memory state. Treat the
                    # packed RAM output as a zero-arrival sequential boundary,
                    # not as a combinational dependency on write data.
                    for state_node in io_nodes['o']:
                        if state_node not in self.nodes:
                            raise ValueError(
                                'memory state output {} is undeclared in {}'.format(
                                    state_node, gate_name
                                )
                            )
                        self.nodes[state_node]['ntype'] = 'input'
                        self.nodes[state_node]['is_module'] = 0
                # deal with arithmetic blocks, whose width may be larger than 1
                elif '.' in sentence:
                    gate_type = gate_type.split('_')[0]
                    node_is_module = 1
                    if gate_type in ('left', 'right'):
                        gate_type = 'mux'
                        node_is_module = 0
                    elif gate_type == 'mult':
                        gate_type = 'add'
                    try:
                        io_nodes = self.get_moduleIOnodes(sentence)
                    except ValueError:
                        if self.structural_audit:
                            self.parser_diagnostics['skipped_malformed_instances'] += 1
                            continue
                        raise

                    # set the node information for the fanout nodes
                    fanout_nodes = io_nodes['o']
                    fanout_nodes.reverse()
                    # if gate_type in ['decoder','encoder'] and len(io_nodes['i'])>=32:
                    #     gate_type = 'input'
                    #     for n in fanout_nodes:
                    #         if self.nodes.get(n, None) is None and 'open' in n:
                    #             self.nodes[n] = {'ntype': gate_type, 'is_po': False, 'is_module': 0}
                    #         else:
                    #             self.nodes[n]['ntype'] = gate_type
                    #             self.nodes[n]['is_module'] = 0
                    #     continue

                    for n in fanout_nodes:
                        if self.nodes.get(n, None) is None and 'open' in n:
                            self.nodes[n] = {
                                'ntype': gate_type,
                                'is_po': False,
                                'is_module': node_is_module,
                            }
                        else:
                            self.nodes[n]['ntype'] = gate_type
                            self.nodes[n]['is_module'] = node_is_module

                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        target_type_map = ntype2id_module if node_is_module else ntype2id_gate
                        target_type_map[gate_type] = target_type_map.get(
                            gate_type, len(target_type_map)
                        )
                        fo2fi[n] = []
                        self.fo2fi[n] = ('module', gate_name,[])


                    width = 0
                    # add the edges between fanin nodes and fanout nodes
                    for port, fanin_nodes in io_nodes.items():
                        for idx,fi in enumerate(fanin_nodes):
                            fo2fi_bit_position[(gate_name,fi)] = len(fanin_nodes) - idx

                            #fanins_bit_position[fi] = len(fanin_nodes) - idx
                        fanin_nodes.reverse()

                        if port == 'o':
                            continue
                        # for port i0,i1: link fi_i[j...0] with fo[j]
                        elif port.startswith('i') and len(fanout_nodes)!=1 and gate_type not in ['encoder','decoder']:

                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes[:i+1])
                                self.fo2fi[fanout_node][2].extend(fanin_nodes[:i+1])
                                self.nodes[fanout_node]['width'] = i+1

                        # for port sel or one output modules, e.g., eq, lt: link all fi_s with each fo
                        else:
                            for i, fanout_node in enumerate(fanout_nodes):
                                fo2fi[fanout_node].extend(fanin_nodes)
                                self.fo2fi[fanout_node][2].extend(fanin_nodes)
                                self.nodes[fanout_node]['width'] = len(fanin_nodes)

                        #width = max(len(fanin_nodes), width)
                # deal with other one-output gates, e.g., or, and...
                else:
                    # get the paramater list
                    close_index = sentence.rfind(')')
                    if close_index <= open_index:
                        raise ValueError('unterminated gate declaration: {!r}'.format(sentence))
                    io_wires = split_top_level(sentence[open_index + 1:close_index])
                    if len(io_wires) < 2:
                        raise ValueError('gate {} has no fanin'.format(gate_name))
                    fanout_nodes = self.parse_wire(io_wires[0])
                    if len(fanout_nodes) != 1:
                        raise ValueError(
                            'primitive gate {} must have one output bit, got {}'.format(
                                gate_name, len(fanout_nodes)
                            )
                        )
                    fanout_node = fanout_nodes[0]
                    fanin_nodes = []
                    for wire in io_wires[1:]:
                        bits = self.parse_wire(wire)
                        if len(bits) != 1:
                            raise ValueError(
                                'primitive gate {} input {!r} is not scalar'.format(
                                    gate_name, wire
                                )
                            )
                        fanin_nodes.append(bits[0])

                    if fanout_node not in self.nodes:
                        raise ValueError('gate output {} is undeclared'.format(fanout_node))

                    if gate_type not in self.buf_types:
                        ntype2id[gate_type] = ntype2id.get(gate_type, len(ntype2id))
                        ntype2id_gate[gate_type] = ntype2id_gate.get(gate_type, len(ntype2id_gate))
                    fo2fi[fanout_node] = fanin_nodes
                    self.fo2fi[fanout_node] = ('gate', gate_name,fanin_nodes)
                    self.nodes[fanout_node]['ntype'] = gate_type

        if self.flag_runtime:
            runtime += time()-start
        if not self.structural_audit:
            if options.parser_constant_impl == 'scan':
                self.fo2fi = simplify_constants_scan(
                    self.fo2fi, self.nodes, gate_func_map
                )
            elif options.parser_constant_impl == 'worklist':
                self.fo2fi = simplify_constants_worklist(
                    self.fo2fi, self.nodes, gate_func_map
                )
            elif options.parser_constant_impl == 'compare':
                scan_nodes = copy.deepcopy(self.nodes)
                worklist_nodes = copy.deepcopy(self.nodes)
                scan_fo2fi = simplify_constants_scan(
                    copy.deepcopy(self.fo2fi), scan_nodes, gate_func_map
                )
                worklist_fo2fi = simplify_constants_worklist(
                    copy.deepcopy(self.fo2fi), worklist_nodes, gate_func_map
                )
                if (
                        list(scan_fo2fi.items()) != list(worklist_fo2fi.items())
                        or scan_nodes != worklist_nodes):
                    raise AssertionError(
                        'constant propagation implementations differ for {}'.format(
                            self.design_name
                        )
                    )
                self.fo2fi = worklist_fo2fi
                self.nodes = worklist_nodes


        # deal with the buffers/NOT gates
        #   record the input-output and output-input pair of buffer/NOT
        is_buf ={}
        for fanout, (fanout_type,_, fanins) in self.fo2fi.items():
            gate_type = self.nodes[fanout]['ntype']
            if gate_type in self.buf_types:
                assert len(fanins)==1
                fanin = fanins[0]
                buf_o2i[fanout] = fanin
                buf_i2o.setdefault(fanin, []).append(fanout)
                is_buf[fanout] = True



        # get the edges and check whether each node is connected or not
        is_linked = {}
        visited_po = {}
        to_duplicate_node = []
        for fanout, (fanout_type, cell_name, fanins) in self.fo2fi.items():

            if fanout not in self.nodes:
                if self.structural_audit:
                    self.parser_diagnostics['skipped_malformed_instances'] += 1
                    continue
                raise KeyError(fanout)

            dst = fanout
            src_list = []

            # add edges from fanin to fanout, consdiering replacement of buffer
            #if self.nodes[fanout]['ntype'] not in self.buf_types:
            if not is_buf.get(fanout,False):
                for fanin in fanins:
                    src = fanin
                    num_inv = 0
                    # for each fanin, recusively backtrace to predecessor until meet non-buffer node
                    while buf_o2i.get(src, None) is not None:
                        src = buf_o2i[src]
                        if src not in self.nodes:
                            if self.structural_audit:
                                break
                            raise KeyError(src)
                        if self.nodes[src]['ntype'] == 'not':
                            num_inv += 1
                    if self.structural_audit and src not in self.nodes:
                        self.parser_diagnostics['skipped_malformed_instances'] += 1
                        continue
                    if self.flag_runtime:
                        start = time()
                    # skip connections between register IO
                    if 'reg' in src and src.endswith('_q') and dst.endswith('_d') and src[:src.rfind('_')]==dst[:dst.rfind('_')]:
                        continue

                    is_inv = num_inv%2
                    bit_position = fo2fi_bit_position.get((cell_name,fanin),None)
                    src_list.append((src,bit_position,is_inv))
                    self.edges.append(
                        (src, dst, {'bit_position': bit_position,'is_inv':is_inv})
                    )
                    is_linked[src] = True
                    is_linked[dst] = True
                    if self.flag_runtime:
                        runtime += time()-start

            # deal with the special condition that the buf output is PO while buf input is PI
            #
            else:
                assert len(fanins) == 1, "{} {}".format(fanout,fanins)
                src = fanins[0]
                while buf_o2i.get(src, None) is not None:
                    src = buf_o2i[src]
                if src not in self.nodes:
                    if self.structural_audit:
                        self.parser_diagnostics['skipped_malformed_instances'] += 1
                        continue
                    raise KeyError(src)

                if self.nodes[fanout]['is_po'] and self.nodes[src]['ntype'] in ["input", "1'b0", "1'b1"]:
                    assert not visited_po.get(fanout,False)
                    self.nodes[fanout]['ntype'] = self.nodes[src]['ntype']
                    self.nodes[fanout]['width'] = self.nodes[src].get('width',0)
                    self.nicknames[fanout] = src
                    visited_po[fanout] = True
                continue

            # Clone the source logic into every PO reached through a buffer tree.
            pending = list(buf_i2o.get(dst, []))
            reachable_pos = []
            visited_buffers = set()
            while pending:
                buffered_dst = pending.pop()
                if buffered_dst in visited_buffers:
                    continue
                visited_buffers.add(buffered_dst)
                if self.nodes[buffered_dst]['is_po']:
                    reachable_pos.append(buffered_dst)
                pending.extend(buf_i2o.get(buffered_dst, []))

            for buffered_po in reachable_pos:
                for (src, bit_pos,is_inv) in src_list:
                    # skip connections between register IO
                    if 'reg' in src and src.endswith('_q') and buffered_po.endswith('_d') and src[:src.rfind('_')]==buffered_po[:buffered_po.rfind('_')]:
                        continue
                    self.edges.append(
                        (src, buffered_po, {'bit_position': bit_pos,'is_inv':is_inv})
                    )

                    is_linked[src] = True
                    is_linked[buffered_po] = True

                self.nodes[buffered_po]['ntype'] = self.nodes[fanout]['ntype']
                self.nodes[buffered_po]['is_module'] = self.nodes[fanout]['is_module']


        # deal with the POs that have successors
        # duplicate the PO, the new node is set as non-po
        # break the coonections between PO and successors
        # add connectins between new nod and successors

        deduplicated_edges = []
        visited = set()
        for src, dst, edict in self.edges:
            edge_identity = (
                src,
                dst,
                edict.get('bit_position'),
                edict.get('is_inv'),
            )
            if edge_identity in visited:
                continue
            visited.add(edge_identity)
            deduplicated_edges.append((src, dst, edict))

        duplicated_pos = {
            src for src, _, _ in deduplicated_edges if self.nodes[src]['is_po']
        }
        for po in duplicated_pos:
            duplicate = '{}_duplicate'.format(po)
            if duplicate in self.nodes:
                raise ValueError('duplicate PO node already exists: {}'.format(duplicate))
            self.nodes[duplicate] = self.nodes[po].copy()
            self.nodes[duplicate]['is_po'] = False
            is_linked[duplicate] = True

        new_edges = []
        for src, dst, edict in deduplicated_edges:
            effective_src = (
                '{}_duplicate'.format(src) if src in duplicated_pos else src
            )
            new_edges.append((effective_src, dst, edict))
            if dst in duplicated_pos:
                new_edges.append(
                    (effective_src, '{}_duplicate'.format(dst), edict)
                )
        self.edges = new_edges


        self.nodes = {n: self.nodes[n] for n in self.nodes.keys() if self.nodes[n]['ntype'] not in ['wire', None]}

        if self.structural_audit:
            # Without timing labels, use the structural output declarations as
            # endpoint candidates. This branch is only for size auditing.
            self.po_label_names = {
                name for name, info in self.nodes.items() if info['is_po']
            }



        # construct the graph
        if self.flag_runtime:
            start = time()
        src_nodes, dst_nodes = [[],[]],[[],[]]
        graph_info = {}
        node2nid = {}
        nid2node = {}
        nodes_type,nodes_value,nodes_delay,nodes_name, nodes_width,POs_label, = [],[],[],[],[],[]
        is_po,is_pi= [],[]
        is_module = []
        for node,node_info in self.nodes.items():
            if not node_info['is_po'] and is_linked.get(node,None) is None:
                continue

            #nodes_name.append((node,node_info.get('nicknames',None)))

            if (node_info['ntype'] in self.buf_types or node_info['ntype']=='output'):
                if self.flag_log: print("removing no-type PO:", node)
                continue


            nid = len(node2nid)
            node2nid[node] = nid
            nid2node[node2nid[node]] = node

            nodes_name.append(node)
            nodes_type.append(node_info['ntype'])


            is_module.append(node_info['is_module'])
            nodes_width.append(node_info.get('width',0))
            if node_info['ntype'] == "1'b0":
                nodes_value.append(0)
            elif node_info['ntype'] == "1'b1":
                nodes_value.append(1)
            else:
                nodes_value.append(2)

            # set the PI delay
            #nodes_delay.append(self.pi_delay.get(node,0))

            flag_pi, flag_po = 0,0
            if self.pi_delay.get(node,None) is not None:
                flag_pi = 1
            if node_info['is_po'] and node in self.po_label_names:
                flag_po = 1
                if node_info['ntype'] in ["input","1'b0","1'b1"]:
                    flag_pi = 1
                    flag_po = 0
                register_base = _REGISTER_VECTOR_SUFFIX_RE.sub('', node)
                if 'reg' in register_base and not register_base.endswith('d'):
                    flag_po = 0
                    #print(node)
            is_pi.append(flag_pi)
            is_po.append(flag_po)

        bit_position = []
        is_inv = [[], []]

        if len(src_nodes[1])==0:
            node2nid['extra1'] = len(node2nid)
            node2nid['extra2'] = len(node2nid)
            src_nodes[1] = [node2nid['extra1']]
            dst_nodes[1] = [node2nid['extra2']]
            is_po.extend([0,0])
            is_pi.extend([0,0])
            is_module.extend([0,0])
            nodes_width.extend([0,0])
            nodes_value.extend([0,0])
            bit_position.extend([0])
            is_inv[1].extend([0])

        nodes_valueOnehot = th.zeros((len(node2nid),3),dtype=th.float)

        for i, v in enumerate(nodes_value):
            nodes_valueOnehot[i][v] = 1

        # get the src_node list and dst_node list
        for eid, (src, dst, edict) in enumerate(self.edges):
            if self.structural_audit and (
                    src not in node2nid or dst not in node2nid
            ):
                self.parser_diagnostics['skipped_malformed_instances'] += 1
                continue
            #print(src,dst)
            if self.flag_log and 'reg' in src and src.endswith('_q') and dst.endswith('_d') and src[:src.rfind('_')] == dst[:dst.rfind('_')]:
                print('###',src, dst)
            if self.flag_log and node2nid.get(src,None) and self.nodes[src]['is_po']:
                print('***', src, dst)

            assert nodes_value[node2nid[dst]] not in [0,1]
            edge_set_idx = is_module[node2nid[dst]]

            if node2nid.get(src,None) is not None:
                src_nid = node2nid[src]
                src_nodes[edge_set_idx].append(src_nid)
                dst_nodes[edge_set_idx].append(node2nid[dst])
                is_inv[edge_set_idx].append(edict['is_inv'])
                if edge_set_idx==1:
                    bit_position.append(edict['bit_position'])


        # for i,d in enumerate(nodes_outdegree):
        #     if d>20:
        #         print(nodes_name[i],d)
        #         exit()
        # exit()

        # print(self.nodes['g36'])


        graph = dgl.heterograph(
        {('node', 'intra_module', 'node'): (th.tensor(src_nodes[1]), th.tensor(dst_nodes[1])),
         ('node', 'intra_gate', 'node'): (th.tensor(src_nodes[0]), th.tensor(dst_nodes[0]))
         },num_nodes_dict={'node':len(node2nid)}
        )

        graph.ndata['is_po'] = th.tensor(is_po)
        graph.ndata['is_pi'] = th.tensor(is_pi)
        graph.ndata['is_module'] = th.tensor(is_module)
        graph.ndata['width'] = th.tensor(nodes_width,dtype=th.float).unsqueeze(1)
        graph.ndata['degree'] = (
            graph.out_degrees(etype='intra_gate')
            + graph.out_degrees(etype='intra_module')
        ).to(dtype=th.float).unsqueeze(1)
        graph.ndata['value'] = nodes_valueOnehot
        graph.edges['intra_module'].data['bit_position'] = th.tensor(bit_position, dtype=th.float)
        graph.edges['intra_module'].data['is_inv'] = th.tensor(is_inv[1], dtype=th.float)
        graph.edges['intra_gate'].data['is_inv'] = th.tensor(is_inv[0], dtype=th.float)

        if self.flag_runtime:
            runtime += time() - start
            print('Runtime for {}:'.format(self.design_name),runtime)

        graph_info['pre_filter_nodes'] = graph.number_of_nodes()
        graph_info['pre_filter_intra_gate_edges'] = graph.number_of_edges('intra_gate')
        graph_info['pre_filter_intra_module_edges'] = graph.number_of_edges('intra_module')

        if self.flag_log or self.flag_runtime:
            print('\t pre-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),
                                                               graph.number_of_edges('intra_gate'),
                                                               graph.number_of_edges('intra_module')))
        # print('\t pre-filter: #node:{}, #edges:{}'.format(graph.number_of_nodes(),
        #                                                       graph.number_of_edges('intra_gate')+graph.number_of_edges('intra_module')))
        # return None, None
        nodes_list = th.tensor(range(graph.number_of_nodes()))
        PIs_nid = nodes_list[graph.ndata['is_pi'] == 1].numpy().tolist()
        PIs_name = [nodes_name[n] for n in PIs_nid]
        POs_nid = nodes_list[graph.ndata['is_po'] == 1].numpy().tolist()
        POs_name = [nodes_name[n] for n in POs_nid]
        graph_info['pre_filter_endpoint_names'] = list(POs_name)
        #print(POs_name)
        # po = 'i183_q_reg_d'
        # pi = 'g42'
        # po_nid = node2nid[pi]
        # cur_nids = Queue()
        # cur_nids.put(po_nid)
        # all = []
        # homo_g = heter2homo(graph)
        #
        # visited = {}
        # while cur_nids.qsize()>0:
        #     cur_nid = cur_nids.get()
        #     if not visited.get(cur_nid,False):
        #         visited[cur_nid] = True
        #
        #         preds = homo_g.successors(cur_nid)
        #         all.append(cur_nid)
        #         print(cur_nid,nodes_name[cur_nid])
        #         for n in preds:
        #             if not visited.get(n,False):
        #                 cur_nids.put(n.item())
        #
        # fanin_nodes = [nodes_name[n] for n in all]
        # print(set(fanin_nodes))
        # # print(set(POs_name).intersection(fanin_nodes))
        # exit()
        #


        #filter out the irrelevant nodes that are not connected to any of the labeled PO
        remain_nodes,remove_nodes = graph_filter(graph)
        remain_nodes = remain_nodes.numpy().tolist()
        graph.remove_nodes(remove_nodes)
        graph.ndata['degree'] = (
            graph.out_degrees(etype='intra_gate')
            + graph.out_degrees(etype='intra_module')
        ).to(dtype=th.float).unsqueeze(1)
        is_module = filter_list(is_module,remain_nodes)
        nodes_type = filter_list(nodes_type,remain_nodes)
        nodes_name = filter_list(nodes_name, remain_nodes)

        nodes_list = th.tensor(range(graph.number_of_nodes()))
        PIs_nid = nodes_list[graph.ndata['is_pi']==1].numpy().tolist()
        PIs_name = [nodes_name[n] for n in PIs_nid]
        POs_nid = nodes_list[graph.ndata['is_po'] == 1].numpy().tolist()
        POs_name = [nodes_name[n] for n in POs_nid]

        nname2nid = {nm:nid for nid,nm in enumerate(nodes_name)}

        # print(PIs_name)
        # exit()
        # print(self.nodes.get('i9661_q_reg_q',None))
        # print(self.nodes.get('Y22_neg[6]', None))
        # print(buf_i2o['i9661_q_reg_q'])
        # print(nname2nid.get('i9661_q_reg_q',None))
        # exit()

        # get the topological level of the PO nodes
        topo = gen_topo(graph)
        nodes_level = th.zeros((len(nodes_name),1),dtype=th.float)
        for l, nodes in enumerate(topo):
            nodes_level[nodes] = l
        graph.ndata['level'] = nodes_level
        POs_level = nodes_level[POs_nid, 0].to(dtype=th.int64).tolist()

        nodes_level = nodes_level.squeeze(1).numpy().tolist()
        graph.edges['intra_gate'].data['rating'] = edges_srcLevel_rating(graph,nodes_level,'intra_gate')
        graph.edges['intra_module'].data['rating'] = edges_srcLevel_rating(graph, nodes_level, 'intra_module')


        # filter out the POs that have abnormal label (large topo level but zero delay)
        remain_pos_idx = []
        for i,level in enumerate(POs_level):
            nid = POs_nid[i]
            PO_name = POs_name[i]
            PO_label = self.po_labels.get(PO_name, -1)

            #if False:
            if (PO_label==0 and level>=2):
            #if (PO_label==0 and level>=2) or (PO_label==1 and level>=5) or (PO_label==2 and level>=10):
                if self.flag_log: print('\t removing PO:',PO_name,PO_label,level)
                graph.ndata['is_po'][nid] = 0
            else:
                remain_pos_idx.append(i)
        POs_level = filter_list(POs_level,remain_pos_idx)
        POs_name = filter_list(POs_name, remain_pos_idx)
        POs_nid = filter_list(POs_nid, remain_pos_idx)

        PIs_delay = []
        for node in PIs_name:
            pi_name = self.nicknames.get(node, node)
            PIs_delay.append(self.pi_delay.get(pi_name, 0) if self.structural_audit else self.pi_delay[pi_name])
        POs_label = []
        for node in POs_name:
            POs_label.append(self.po_labels.get(node, -1))

        #print(graph.edges(etype='intra_module'))
        # src_m,dst_m = graph.edges(etype='intra_module')
        # src_m=src_m.numpy().tolist()
        # dst_m = dst_m.numpy().tolist()
        # visited = {}
        # for i,(src,dst) in enumerate(list(zip(src_m,dst_m))):
        #     if visited.get(dst,False):
        #         continue
        #     bit_pos = graph.edges['intra_module'].data['bit_position'][i].item()
        #     src_name = nodes_name[src]
        #     dst_name = nodes_name[dst]
        #
            # if bit_pos>32 and nodes_type[dst] in ['decoder','encoder']:
            #     visited[dst] = True
            #     print('+++++',bit_pos,src_name,dst_name)

                #exit()

        # save the necessary graph information
        graph_info['topo'] = topo
        graph_info['ntype'] = nodes_type
        graph_info['nodes_name'] = nodes_name
        graph_info['nname2nid'] = nname2nid
        #graph_info['POs'] = POs
        #graph_info['POs_label'] = th.tensor(POs_label,dtype=th.float)
        graph_info['POs_level_max'] = th.tensor(POs_level,dtype=th.float)
        graph_info['POs_name'] = POs_name
        graph_info['PIs_name'] = PIs_name
        graph_info['post_filter_nodes'] = graph.number_of_nodes()
        graph_info['post_filter_endpoint_names'] = list(POs_name)
        graph_info['design_name'] = self.design_name
        graph_info['nicknames'] = self.nicknames
        graph_info['PIs_delay'] = PIs_delay
        graph_info['POs_label'] = POs_label
        graph_info['parser_diagnostics'] = dict(self.parser_diagnostics)

        if self.flag_log or self.flag_runtime:
            print('\t post-filter: #node:{}, #edges:{}, {}'.format(graph.number_of_nodes(),graph.number_of_edges('intra_gate'),graph.number_of_edges('intra_module')))
            print('\t #PO:{}'.format(len(POs_name)))

        # print(POs_name)
        # exit()
        # print('\t',graph_info)

        return graph, graph_info

    def parse(self):
        #self.pi_delay,self.po_labels,_ = parse_golden(os.path.join(self.design_path,'golden_0.txt'))
        self.pi_delay, self.po_labels, _ = parse_golden(self.golden_file)
        # for p, d in self.pi_delay.items():
        #     assert d == 0, print("base case with non-zero input delay: {} {}".format(p, d))
        if self.pi_delay is None:
            return None,None
        if self.po_label_names is None:
            self.po_label_names = set(self.po_labels)
        else:
            self.po_label_names.update(self.po_labels)
        graph, graph_info = self.parse_verilog()
        if graph is None:
            return None,None
        graph_info['base_po_labels'] = {
            name: self.po_labels.get(name, -1)
            for name in self.po_label_names
        }

        return graph,graph_info

    def parse_structural_audit(self):
        """Parse a netlist without timing labels for structural size auditing."""
        self.structural_audit = True
        self.pi_delay = {}
        self.po_labels = {}
        self.po_label_names = set()
        graph, graph_info = self.parse_verilog()
        if graph is None:
            return None, None
        graph_info['base_po_labels'] = [-1] * len(graph_info['POs_name'])
        graph_info['audit_mode'] = 'structural_output_endpoints'
        return graph, graph_info

def parse_golden(file_path):
    return parse_golden_file(file_path, keep_all_paths=False)


def parse_golden_new(file_path):
    return parse_golden_file(file_path, keep_all_paths=True)


def validate_ntype_map(type_map, name):
    if not isinstance(type_map, dict):
        raise ValueError('{} must be a dictionary'.format(name))
    ids = sorted(type_map.values())
    if ids != list(range(len(ids))):
        raise ValueError('{} IDs must be contiguous from zero, got {}'.format(name, ids))


def load_ntype_maps():
    global ntype2id, ntype2id_gate, ntype2id_module
    if not os.path.exists(ntype_file):
        return
    with open(ntype_file, 'rb') as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, (tuple, list)) or len(loaded) != 3:
        raise ValueError('invalid ntype schema in {}'.format(ntype_file))
    for name, type_map in zip(
            ('ntype2id', 'ntype2id_gate', 'ntype2id_module'), loaded):
        validate_ntype_map(type_map, name)
    ntype2id, ntype2id_gate, ntype2id_module = [dict(type_map) for type_map in loaded]


def atomic_pickle_dump(value, file_path):
    temporary_path = '{}.tmp.{}'.format(file_path, os.getpid())
    try:
        with open(temporary_path, 'wb') as handle:
            pickle.dump(value, handle)
        os.replace(temporary_path, file_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    global ntype2id,ntype2id_gate,ntype2id_module
    if options.parser_workers < 1:
        raise ValueError('--parser_workers must be at least 1')
    load_ntype_maps()
    dataset = []
    designs_group = {0: [], 1: [], 2: [], 3: []}
    rng = random.Random(options.parser_seed)

    if 'round6' in rawdata_path:
        subdirs = sorted(
            name for name in os.listdir(rawdata_path)
            if os.path.isdir(os.path.join(rawdata_path, name))
        )
    else:
        subdirs = ['']
    for subdir in subdirs:
        subdir_path = os.path.join(rawdata_path,subdir)
        for design in sorted(os.listdir(subdir_path)):
            #if '308' not in design: continue

            # if 'round6' in rawdata_path and int(design.split('_')[-1]) in [110,220,183,185,319,320,329,371,383,392,399]:
            #     continue
            
            #if design  in [ 'ldpcenc', 'systemcaes','sha3', 'wb_conmax','oc_wb_dma','mc6809', 's15850', 'tv80', 'oc_mem_ctrl','ecg','y_dct']: continue


            #if design in ['sin','multiplier','div','sqrt','mem_ctrl','log2','y_huff','voter']: continue
            #if design not in ['priority', 'adder', 'max', 'square',  'router', 'int2float', 'cavlc', 'dec', 'arbiter', 'bar']: continue

            # if not flag and design!='arbiter': continue
            # flag=True
            # if design=='voter':continue

            design_dir = os.path.join(subdir_path,design)
            if not os.path.isdir(design_dir):
                continue
            base_golden_file = os.path.join(
                design_dir, '{}_0'.format(design), 'golden.txt'
            )
            if not os.path.isfile(base_golden_file):
                continue
            print("-----Parsing {}-----".format(design))
            netlist_candidates = [
                os.path.join(design_dir, '{}.v'.format(design)),
                os.path.join(design_dir, '{}_case.v'.format(design)),
            ]
            netlist_file = next(
                (path for path in netlist_candidates if os.path.isfile(path)), None
            )
            if netlist_file is None:
                raise FileNotFoundError(
                    'no netlist found for {} (tried {})'.format(
                        design, ', '.join(netlist_candidates)
                    )
                )

            case_pattern = re.compile(r'^{}_(\d+)$'.format(re.escape(design)))
            case_indexs = []
            for entry in sorted(os.listdir(design_dir)):
                match = case_pattern.match(entry)
                golden_path = os.path.join(design_dir, entry, 'golden.txt')
                if match is not None and os.path.isfile(golden_path):
                    case_indexs.append(int(match.group(1)))
            case_indexs.sort()
            case_tasks = [
                (
                    idx,
                    os.path.join(
                        design_dir, '{}_{}'.format(design, idx), 'golden.txt'
                    ),
                )
                for idx in case_indexs
            ]
            po_label_names = set()
            for _, golden_path in case_tasks:
                po_label_names.update(parse_golden_endpoint_names(golden_path))

            flag_log = False
            parser = Parser(
                netlist_file,
                base_golden_file,
                flag_log,
                po_label_names=po_label_names,
            )

            graph, graph_info = parser.parse()
            if graph is None or th.sum(graph.ndata['is_po']).item()==0:
                continue



            # if 'round7' in design_dir and len(graph_info['POs_name'])<10:
            #     continue

            graph_info['delay-label_pairs'] = []
            graph_info.setdefault('parser_diagnostics', {})
            graph_info['parser_diagnostics'].update({
                'dropped_critical_pis': 0,
                'dropped_critical_pos': 0,
                'skipped_cases': 0,
                'missing_po_labels': 0,
            })
            graph_info['case_indices'] = []
            valid_pis = set(
                th.nonzero(graph.ndata['is_pi'] == 1, as_tuple=False).squeeze(1).tolist()
            )
            valid_pos = set(
                th.nonzero(graph.ndata['is_po'] == 1, as_tuple=False).squeeze(1).tolist()
            )
            endpoint_name_to_nid = dict(graph_info['nname2nid'])
            for node_name, nickname in graph_info['nicknames'].items():
                node_id = graph_info['nname2nid'].get(node_name)
                if node_id is not None:
                    endpoint_name_to_nid.setdefault(nickname, node_id)
            pi_names = [
                graph_info['nicknames'].get(node, node)
                for node in graph_info['PIs_name']
            ]

            case_context = {
                'pi_names': pi_names,
                'po_names': graph_info['POs_name'],
                'base_po_labels': graph_info['base_po_labels'],
                'endpoint_name_to_nid': endpoint_name_to_nid,
                'valid_pis': valid_pis,
                'valid_pos': valid_pos,
                'design_name': design,
            }
            worker_pool = None
            worker_count = min(options.parser_workers, len(case_tasks))
            if not case_tasks:
                case_results = ()
            elif worker_count == 1:
                initialize_case_parser(case_context)
                case_results = map(parse_case_task, case_tasks)
            else:
                start_method = (
                    'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
                )
                worker_pool = mp.get_context(start_method).Pool(
                    worker_count,
                    initializer=initialize_case_parser,
                    initargs=(case_context,),
                )
                case_results = worker_pool.imap(
                    parse_case_task, case_tasks, chunksize=1
                )

            try:
                for (
                        idx,
                        PIs_delay,
                        POs_label,
                        POs_label_residual,
                        pi2po_edges,
                        dropped_pis,
                        dropped_pos,
                ) in case_results:
                    if PIs_delay is None:
                        graph_info['parser_diagnostics']['skipped_cases'] += 1
                        continue
                    graph_info['parser_diagnostics']['dropped_critical_pis'] += dropped_pis
                    graph_info['parser_diagnostics']['dropped_critical_pos'] += dropped_pos
                    graph_info['parser_diagnostics']['missing_po_labels'] += sum(
                        label < 0 for label in POs_label
                    )

                    labeled_po_nids = {
                        endpoint_name_to_nid[name]
                        for name, label in zip(graph_info['POs_name'], POs_label)
                        if label >= 0
                    }

                    if pi2po_edges is None:
                        nodes_delay = th.zeros(
                            (graph.number_of_nodes(), 1), dtype=th.float
                        )
                        nodes_delay[graph.ndata['is_pi'] == 1] = th.tensor(
                            PIs_delay, dtype=th.float
                        ).unsqueeze(1)
                        graph.ndata['delay'] = nodes_delay
                        pi2po_edges = get_pi2po_edges(graph, graph_info)
                    pi2po_edges = filter_pi2po_edges_by_destinations(
                        pi2po_edges, labeled_po_nids
                    )

                    if validate_pi2po_edges(pi2po_edges) == 0:
                        graph_info['parser_diagnostics']['skipped_cases'] += 1
                        continue
                    graph_info['delay-label_pairs'].append((
                        PIs_delay,
                        POs_label,
                        POs_label_residual,
                        pi2po_edges,
                    ))
                    graph_info['case_indices'].append(idx)
            finally:
                if worker_pool is not None:
                    worker_pool.close()
                    worker_pool.join()


            POs_base_label = []
            for node in graph_info['POs_name']:
                POs_base_label.append(graph_info['base_po_labels'].get(node, -1))
            graph_info['base_po_labels'] = POs_base_label

            if len(graph_info['delay-label_pairs']) < options.min_cases_per_design:
                continue

            if 'round7' in design_dir:
                if len(graph_info['delay-label_pairs'][0][1]) <= 150:
                    continue
                if graph_info['design_name'] in ['s15850', 's5378', 'tv80', 'sha3', 'ldpcenc', 'mc6809']: continue

            if graph is not None:
                dataset.append((graph,graph_info))

            print(design, '#pairs', len(graph_info['delay-label_pairs']))
            diagnostics = graph_info['parser_diagnostics']
            if any(diagnostics.values()):
                print('{} parser diagnostics: {}'.format(design, diagnostics))
            if flag_group:
                group_id = int(int(design.split('_')[-1]) / 100)
            else:
                group_id = 0
            designs_group.setdefault(group_id, []).append(graph_info['design_name'])

            # print(graph.ndata['is_pi'])
            # print(graph_info['delay-label_pairs'])
        # exit()

    for name, type_map in (
        ('ntype2id', ntype2id),
        ('ntype2id_gate', ntype2id_gate),
        ('ntype2id_module', ntype2id_module),
    ):
        validate_ntype_map(type_map, name)
    atomic_pickle_dump((ntype2id, ntype2id_gate, ntype2id_module), ntype_file)
    print('ntypes:',ntype2id,ntype2id_gate,ntype2id_module)



    final_dataset = []
    for graph, graph_info in dataset:
        is_module = graph.ndata['is_module'].numpy().tolist()
        ntype_onehot = th.zeros((graph.number_of_nodes(), len(ntype2id)), dtype=th.float)
        ntype_onehot_module = th.zeros((graph.number_of_nodes(), len(ntype2id_module)), dtype=th.float)
        ntype_onehot_gate = th.zeros((graph.number_of_nodes(), len(ntype2id_gate)), dtype=th.float)

        for nid, type in enumerate(graph_info['ntype']):
            if type not in ntype2id:
                raise ValueError('node type {!r} is absent from ntype2id'.format(type))
            ntype_onehot[nid][ntype2id[type]] = 1
            if is_module[nid] == 1:
                if type not in ntype2id_module:
                    raise ValueError(
                        'module node type {!r} is absent from ntype2id_module'.format(type)
                    )
                ntype_onehot_module[nid][ntype2id_module[type]] = 1
            else:
                if type not in ntype2id_gate:
                    raise ValueError(
                        'gate node type {!r} is absent from ntype2id_gate'.format(type)
                    )
                ntype_onehot_gate[nid][ntype2id_gate[type]] = 1
        graph.ndata['ntype'] = ntype_onehot
        graph.ndata['ntype_module'] = ntype_onehot_module
        graph.ndata['ntype_gate'] = ntype_onehot_gate
        final_dataset.append((graph,graph_info))

    dataset = final_dataset
    rng.shuffle(dataset)
    print(len(dataset))

    if flag_group:
        for group_id in sorted(designs_group):
            rng.shuffle(designs_group[group_id])

    split_list = {'train': [], 'test': [], 'val': []}
    if flag_split:
        split_point = [0.7,0.8]
    else:
        split_point = [0,0]
    
    for group_id, designs in designs_group.items():
        num_designs = len(designs)
        split_list['train'].extend(designs[:int(split_point[0] * num_designs)])
        split_list['val'].extend(designs[int(split_point[0] * num_designs):int(split_point[1] * num_designs)])
        split_list['test'].extend(designs[int(split_point[1] * num_designs):])

    print('#train:{}, #val:{}, #test:{}'.format(len(split_list['train']), len(split_list['val']), len(split_list['test'])))

    #print(split_list['val'])
    atomic_pickle_dump(split_list, os.path.join(data_savepath, 'split.pkl'))
    atomic_pickle_dump(dataset, os.path.join(data_savepath, 'data.pkl'))
    # with open(os.path.join(data_savepath,'graph.pkl'),'wb') as f:
    #     pickle.dump(final_dataset,f)

if __name__ == "__main__":
    stdout_f = os.path.join(data_savepath,'stdout.log')
    stderr_f = os.path.join(data_savepath, 'stderr.log')

    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        main()
