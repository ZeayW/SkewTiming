import copy
import pathlib
import random
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from parser_helpers import (  # noqa: E402
    build_critical_edges,
    filter_pi2po_edges_by_destinations,
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
    split_verilog_statements,
    validate_pi2po_edges,
)


class ParserHelpersTest(unittest.TestCase):
    @staticmethod
    def constant_gate_functions():
        def xor(values):
            total = sum(values)
            return 2 if total == -2 else 3 if total == -1 else total % 2

        def xnor(values):
            total = sum(values)
            return 3 if total == -2 else 2 if total == -1 else (total + 1) % 2

        def or_gate(values):
            total = sum(values)
            return 2 if total == -2 else 1 if total == -1 else int(total >= 1)

        def and_gate(values):
            total = sum(values)
            return 0 if total == -2 else 2 if total == -1 else int(total == 2)

        def not_gate(values):
            return (values[0] + 1) % 2

        def buf_gate(values):
            return values[0]

        return {
            'xor': xor,
            'xnor': xnor,
            'or': or_gate,
            'and': and_gate,
            'not': not_gate,
            'buf': buf_gate,
        }

    def test_statement_split_is_newline_and_comment_independent(self):
        content = (
            'module top;\r\n'
            '// ignored ; declaration\r\n'
            'input\t a ; \r\n'
            'wire b; /* ignored ; block */\n'
            'assign b=a;\n'
            'endmodule\n'
        )
        self.assertEqual(
            split_verilog_statements(content),
            ['module top', 'input\t a', 'wire b', 'assign b=a', 'endmodule'],
        )

    def test_declarations_support_qualifiers_multiple_names_and_both_ranges(self):
        self.assertEqual(
            parse_declaration('input wire signed [7:0] data, mask'),
            ('input', (7, 0), ['data', 'mask']),
        )
        self.assertEqual(
            parse_declaration('output logic [0:3] result'),
            ('output', (0, 3), ['result']),
        )
        self.assertEqual(inclusive_range(7, 4), [7, 6, 5, 4])
        self.assertEqual(inclusive_range(0, 3), [0, 1, 2, 3])

    def test_named_ports_keep_concatenations_intact(self):
        ports = parse_named_ports(
            "mux_4 m0 (.i0({2'b00, a, b}), .i1(c), .sel(s), .o(out))"
        )
        self.assertEqual(
            ports,
            {'i0': "{2'b00, a, b}", 'i1': 'c', 'sel': 's', 'o': 'out'},
        )

    def test_parameterized_instance_uses_actual_port_list(self):
        statement = (
            "read_port_1_32_1_0_31_0 #("
            ".INIT_VALUE(64'b0), .RAM_INIT_STATE(\"2\")) memory_read "
            "(.ra(addr), .ram(memory), .re(1'b1), .rd(data))"
        )
        self.assertEqual(
            parse_instance_header(statement),
            ('read_port_1_32_1_0_31_0', 'memory_read'),
        )
        self.assertEqual(
            parse_named_ports(statement),
            {'ra': 'addr', 'ram': 'memory', 're': "1'b1", 'rd': 'data'},
        )
        self.assertEqual(statement[find_instance_port_open(statement)], '(')
        self.assertTrue(
            statement[find_instance_port_open(statement) + 1:].startswith('.ra')
        )

    def test_legacy_structural_recovery_keeps_final_complete_instance(self):
        malformed = (
            '.i0(a), .i1(b), .sel(s), '
            'AL_MUX m0 (.i0(a), .i1(b), .sel(s), .o(y))'
        )
        self.assertEqual(
            recover_structural_instance(malformed),
            'AL_MUX m0 (.i0(a), .i1(b), .sel(s), .o(y))',
        )
        self.assertIsNone(
            recover_structural_instance('.i0(a), .sel(s), .o(y))')
        )

    def test_unsigned_resize_zero_extends_and_lsb_truncates(self):
        self.assertEqual(
            resize_unsigned_bits(['a', 'b'], 4, lambda: '0'),
            ['0', '0', 'a', 'b'],
        )
        self.assertEqual(
            resize_unsigned_bits(['a', 'b', 'c', 'd'], 2, lambda: '0'),
            ['c', 'd'],
        )

    def test_missing_critical_pi_keeps_weights_aligned(self):
        critical = {'out': [('a', 9), ('missing', 8), ('b', 7)]}
        edges, missing_pis, missing_pos = build_critical_edges(
            critical,
            {'a': 1, 'b': 2, 'out': 3},
            valid_pis={1, 2},
            valid_pos={3},
        )
        self.assertEqual(edges, ([1, 2], [3, 3], [9, 7]))
        self.assertEqual(missing_pis, ['missing'])
        self.assertEqual(missing_pos, [])
        self.assertEqual(validate_pi2po_edges(edges), 2)

    def test_mismatched_pi2po_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mismatched lengths'):
            validate_pi2po_edges(([1], [2], [3, 4]))

    def test_golden_parser_accepts_crlf_and_general_whitespace(self):
        content = (
            '// inputs\r\n'
            'a\t1\r\n'
            'b   2\r\n'
            '// pin to pin level synthesised\r\n'
            'out,   a,  5\r\n'
            'out, b, 7\r\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'golden.txt'
            path.write_bytes(content.encode('utf-8'))
            pi_delay, po_labels, all_paths = parse_golden_file(path, True)
            _, _, critical_paths = parse_golden_file(path, False)

        self.assertEqual(pi_delay, {'a': 1, 'b': 2})
        self.assertEqual(po_labels, {'out': 7})
        self.assertEqual(all_paths, {'out': [('a', 5), ('b', 7)]})
        self.assertEqual(critical_paths, {'out': [('b', 1)]})

    def test_endpoint_scan_returns_every_labeled_endpoint(self):
        content = (
            'a 1\n'
            '// pin to pin level synthesised\n'
            'out, a, 5\n'
            'other, 6\n'
            'clock_only, clk, 9\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'golden.txt'
            path.write_text(content)
            endpoints = parse_golden_endpoint_names(path)

        self.assertEqual(endpoints, {'out', 'other'})

    def test_case_task_preserves_case_arrays_and_grouped_edge_order(self):
        content = (
            'a 1\n'
            'b 2\n'
            '// pin to pin level synthesised\n'
            'out, a, 5\n'
            'other, b, 6\n'
            'out, missing, 8\n'
        )
        context = {
            'base_label_names': {'out', 'other'},
            'pi_names': ['a', 'b'],
            'po_names': ['out', 'other'],
            'base_po_labels': {'out': 5, 'other': 6},
            'endpoint_name_to_nid': {'a': 1, 'b': 2, 'out': 3, 'other': 4},
            'valid_pis': {1, 2},
            'valid_pos': {3, 4},
            'design_name': 'tiny',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'golden.txt'
            path.write_text(content)
            initialize_case_parser(context)
            result = parse_case_task((7, path))

        self.assertEqual(result[0:4], (7, [1, 2], [8, 6], [3, 0]))
        self.assertEqual(result[4], ([1, 2], [3, 4], [5, 6]))
        self.assertEqual(result[5:], (1, 0))

    def test_case_task_fills_missing_union_endpoint_with_minus_one(self):
        content = (
            'a 1\n'
            'b 2\n'
            '// pin to pin level synthesised\n'
            'out, a, 8\n'
        )
        context = {
            'pi_names': ['a', 'b'],
            'po_names': ['out', 'other'],
            'base_po_labels': {'out': 5, 'other': 6},
            'endpoint_name_to_nid': {'a': 1, 'b': 2, 'out': 3, 'other': 4},
            'valid_pis': {1, 2},
            'valid_pos': {3, 4},
            'design_name': 'tiny',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'golden.txt'
            path.write_text(content)
            initialize_case_parser(context)
            result = parse_case_task((9, path))

        self.assertEqual(result[0:4], (9, [1, 2], [8, -1], [3, -1]))
        self.assertEqual(result[4], ([1], [3], [8]))

    def test_pi2po_filter_removes_unlabeled_destinations(self):
        edges = ([1, 2, 1], [3, 4, 5], [8, 9, 7])
        self.assertEqual(
            filter_pi2po_edges_by_destinations(edges, {3, 5}),
            ([1, 1], [3, 5], [8, 7]),
        )

    def test_constant_worklist_matches_scan_on_shuffled_dags(self):
        rng = random.Random(7)
        gate_functions = self.constant_gate_functions()
        for _ in range(20):
            nodes = {
                "1'b0": {'ntype': "1'b0"},
                "1'b1": {'ntype': "1'b1"},
            }
            available = ["1'b0", "1'b1"]
            for index in range(5):
                name = 'input{}'.format(index)
                nodes[name] = {'ntype': 'input'}
                available.append(name)

            records = []
            for index in range(40):
                fanout = 'gate{}'.format(index)
                gate_type = rng.choice(('xor', 'xnor', 'or', 'and'))
                fanins = rng.sample(available, 2)
                nodes[fanout] = {'ntype': gate_type}
                records.append((fanout, ('gate', fanout, fanins)))
                available.append(fanout)
            rng.shuffle(records)
            fo2fi = dict(records)

            scan_nodes = copy.deepcopy(nodes)
            worklist_nodes = copy.deepcopy(nodes)
            scan_result = simplify_constants_scan(
                copy.deepcopy(fo2fi), scan_nodes, gate_functions
            )
            worklist_result = simplify_constants_worklist(
                copy.deepcopy(fo2fi), worklist_nodes, gate_functions
            )
            self.assertEqual(list(worklist_result.items()), list(scan_result.items()))
            self.assertEqual(worklist_nodes, scan_nodes)



if __name__ == '__main__':
    unittest.main()
