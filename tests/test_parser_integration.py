import importlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest

import torch


SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
DGL_AVAILABLE = importlib.util.find_spec('dgl') is not None


@unittest.skipUnless(DGL_AVAILABLE, 'DGL is required for parser graph integration tests')
class ParserIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser_module = importlib.import_module('parser')
        cls.parser_graph_utils = importlib.import_module('parser_graph_utils')

    def test_edge_source_level_ratings_match_per_destination_rank(self):
        src = torch.tensor([0, 1, 2, 3, 2, 0])
        dst = torch.tensor([4, 4, 4, 4, 5, 5])
        levels = torch.tensor([0, 2, 2, 1, 3, 4])
        ratings = self.parser_graph_utils.edge_source_level_ratings(
            src, dst, levels
        )
        torch.testing.assert_close(
            ratings,
            torch.tensor([[3.0], [1.0], [1.0], [2.0], [1.0], [2.0]]),
            rtol=0,
            atol=0,
        )

    def test_po_with_successor_is_preserved_and_degree_is_exact(self):
        netlist = (
            'module tiny;\r\n'
            'input a ;\r\n'
            'input\tb;\r\n'
            'output z;\r\n'
            'output y;\r\n'
            'and g0 (z, a, b);\r\n'
            'not g1 (y, z);\r\n'
            'endmodule\r\n'
        )
        golden = (
            'a 0\r\n'
            'b 0\r\n'
            '// pin to pin level synthesised\r\n'
            'z, a, 1\r\n'
            'y, a, 2\r\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'tiny.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        self.assertIn('z', graph_info['POs_name'])
        self.assertIn('y', graph_info['POs_name'])
        self.assertIn('z_duplicate', graph_info['nodes_name'])
        self.assertEqual(int(graph.ndata['is_po'].sum()), 2)

        z_nid = graph_info['nname2nid']['z']
        duplicate_nid = graph_info['nname2nid']['z_duplicate']
        self.assertEqual(graph.out_degrees(z_nid, etype='intra_gate'), 0)
        self.assertEqual(graph.out_degrees(duplicate_nid, etype='intra_gate'), 1)

        actual_degree = (
            graph.out_degrees(etype='intra_gate')
            + graph.out_degrees(etype='intra_module')
        ).to(torch.float).unsqueeze(1)
        torch.testing.assert_close(graph.ndata['degree'], actual_degree, rtol=0, atol=0)

    def test_structural_audit_parses_without_golden_labels(self):
        netlist = (
            'module tiny;\n'
            'input a;\n'
            'input b;\n'
            'output z;\n'
            'output y;\n'
            'and g0 (z, a, b);\n'
            'not g1 (y, z);\n'
            'endmodule\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'tiny.v'
            netlist_path.write_text(netlist, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), None
            ).parse_structural_audit()

        self.assertEqual(graph_info['audit_mode'], 'structural_output_endpoints')
        self.assertEqual(graph_info['pre_filter_nodes'], 5)
        self.assertEqual(set(graph_info['pre_filter_endpoint_names']), {'z', 'y'})
        self.assertEqual(graph_info['pre_filter_nodes'], graph_info['post_filter_nodes'])
        self.assertEqual(int(graph.ndata['is_po'].sum()), 2)

    def test_parse_wire_preserves_declared_vector_direction(self):
        parser = self.parser_module.Parser('tiny.v', 'golden.txt')
        parser.wires_width = {'descending': (3, 0), 'ascending': (0, 3)}
        self.assertEqual(
            parser.parse_wire('descending'),
            ['descending[3]', 'descending[2]', 'descending[1]', 'descending[0]'],
        )
        self.assertEqual(
            parser.parse_wire('ascending'),
            ['ascending[0]', 'ascending[1]', 'ascending[2]', 'ascending[3]'],
        )

    def test_buffer_fanout_preserves_every_labeled_po(self):
        netlist = (
            'module fanout;\n'
            'input a;\n'
            'input b;\n'
            'wire core;\n'
            'output y;\n'
            'output z;\n'
            'and g0 (core, a, b);\n'
            'buf b0 (y, core);\n'
            'buf b1 (z, core);\n'
            'endmodule\n'
        )
        golden = (
            'a 0\n'
            'b 0\n'
            '// pin to pin level synthesised\n'
            'y, a, 1\n'
            'z, b, 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'fanout.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        self.assertEqual(set(graph_info['POs_name']), {'y', 'z'})
        self.assertEqual(int(graph.ndata['is_po'].sum()), 2)

    def test_po_union_keeps_endpoint_missing_from_baseline(self):
        netlist = (
            'module endpoint_union;\n'
            'input a;\n'
            'input b;\n'
            'output y;\n'
            'output z;\n'
            'and g0 (y, a, b);\n'
            'or g1 (z, a, b);\n'
            'endmodule\n'
        )
        baseline = (
            'a 0\n'
            'b 0\n'
            '// pin to pin level synthesised\n'
            'y, a, 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'endpoint_union.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(baseline, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path),
                str(golden_path),
                po_label_names={'y', 'z'},
            ).parse()

        self.assertEqual(set(graph_info['POs_name']), {'y', 'z'})
        labels_by_name = dict(zip(graph_info['POs_name'], graph_info['POs_label']))
        self.assertEqual(labels_by_name, {'y': 1, 'z': -1})
        self.assertEqual(graph_info['base_po_labels']['z'], -1)
        self.assertEqual(int(graph.ndata['is_po'].sum()), 2)

    def test_mux_uses_unsigned_extension_instead_of_cyclic_repetition(self):
        netlist = (
            'module mux_width;\n'
            'input a;\n'
            'input sel;\n'
            'output [3:0] out;\n'
            "mux_4 m0 (.i0(a), .i1(4'b1111), .sel(sel), .o(out));\n"
            'endmodule\n'
        )
        golden = (
            'a 0\n'
            'sel 0\n'
            '// pin to pin level synthesised\n'
            'out[3], a, 1\n'
            'out[2], a, 1\n'
            'out[1], a, 1\n'
            'out[0], a, 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'mux_width.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        def predecessor_names(node_name):
            node_id = graph_info['nname2nid'][node_name]
            predecessors = graph.predecessors(node_id, etype='intra_gate').tolist()
            return {graph_info['nodes_name'][node] for node in predecessors}

        self.assertNotIn('a', predecessor_names('out[3]'))
        self.assertNotIn('a', predecessor_names('out[2]'))
        self.assertNotIn('a', predecessor_names('out[1]'))
        self.assertIn('a', predecessor_names('out[0]'))

    def test_packed_memory_read_connects_each_output_to_its_ram_stride(self):
        netlist = (
            'module memory_read;\n'
            'input ra;\n'
            'input re;\n'
            'input [3:0] ram;\n'
            'output [1:0] rd;\n'
            'read_port_2_2 helper '
            '(.ra(ra), .ram(ram), .re(re), .rd(rd));\n'
            'endmodule\n'
        )
        golden = (
            'ra 0\n'
            're 0\n'
            'ram[3] 0\n'
            'ram[2] 0\n'
            'ram[1] 0\n'
            'ram[0] 0\n'
            '// pin to pin level synthesised\n'
            'rd[1], ram[3], 1\n'
            'rd[0], ram[2], 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'memory_read.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        def predecessor_names(node_name):
            node_id = graph_info['nname2nid'][node_name]
            predecessors = graph.predecessors(node_id, etype='intra_gate').tolist()
            return {graph_info['nodes_name'][node] for node in predecessors}

        self.assertEqual(
            predecessor_names('rd[0]'),
            {'ram[0]', 'ram[2]', 'ra', 're'},
        )
        self.assertEqual(
            predecessor_names('rd[1]'),
            {'ram[1]', 'ram[3]', 'ra', 're'},
        )

    def test_parameterized_packed_memory_read_uses_instance_ports(self):
        netlist = (
            'module memory_read;\n'
            'input ra;\n'
            'input re;\n'
            'input [3:0] ram;\n'
            'output [1:0] rd;\n'
            'read_port_2_2 #(.INIT_VALUE(4\'b0), .RAM_INIT_STATE("2")) helper '
            '(.ra(ra), .ram(ram), .re(re), .rd(rd));\n'
            'endmodule\n'
        )
        golden = (
            'ra 0\n'
            're 0\n'
            'ram[3] 0\n'
            'ram[2] 0\n'
            'ram[1] 0\n'
            'ram[0] 0\n'
            '// pin to pin level synthesised\n'
            'rd[1], ram[3], 1\n'
            'rd[0], ram[2], 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'memory_read.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        rd0 = graph_info['nname2nid']['rd[0]']
        predecessors = graph.predecessors(rd0, etype='intra_gate').tolist()
        predecessor_names = {graph_info['nodes_name'][node] for node in predecessors}
        self.assertEqual(predecessor_names, {'ram[0]', 'ram[2]', 'ra', 're'})

    def test_scalar_memory_state_is_shared_by_all_read_bits(self):
        netlist = (
            'module memory_read;\n'
            'input ra;\n'
            'input re;\n'
            'input ram;\n'
            'output [1:0] rd;\n'
            'read_port_1_2 helper '
            '(.ra(ra), .ram(ram), .re(re), .rd(rd));\n'
            'endmodule\n'
        )
        golden = (
            'ra 0\n'
            're 0\n'
            'ram 0\n'
            '// pin to pin level synthesised\n'
            'rd[1], ram, 1\n'
            'rd[0], ram, 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'memory_read.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        def predecessor_names(node_name):
            node_id = graph_info['nname2nid'][node_name]
            predecessors = graph.predecessors(node_id, etype='intra_gate').tolist()
            return {graph_info['nodes_name'][node] for node in predecessors}

        self.assertEqual(predecessor_names('rd[0]'), {'ram', 'ra', 're'})
        self.assertEqual(predecessor_names('rd[1]'), {'ram', 'ra', 're'})

    def test_legacy_malformed_instances_are_recovered_or_skipped_with_diagnostics(self):
        netlist = (
            'module malformed;\n'
            'input a;\n'
            'input b;\n'
            'wire core;\n'
            'wire unused;\n'
            'output y;\n'
            '.i0(a), .i1(b), .sel(a), .o(unused));\n'
            '.i0(a), .i1(b), .sel(a), '
            'AL_MUX m0 (.i0(a), .i1(b), .sel(a), .o(core));\n'
            'buf g0 (y, core);\n'
            'endmodule\n'
        )
        golden = (
            'a 0\n'
            'b 0\n'
            '// pin to pin level synthesised\n'
            'y, a, 1\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            netlist_path = pathlib.Path(directory) / 'malformed.v'
            golden_path = pathlib.Path(directory) / 'golden.txt'
            netlist_path.write_text(netlist, encoding='utf-8')
            golden_path.write_text(golden, encoding='utf-8')
            graph, graph_info = self.parser_module.Parser(
                str(netlist_path), str(golden_path)
            ).parse()

        self.assertIsNotNone(graph)
        self.assertIn('y', graph_info['POs_name'])
        self.assertEqual(
            graph_info['parser_diagnostics']['recovered_malformed_instances'], 1
        )
        self.assertEqual(
            graph_info['parser_diagnostics']['skipped_malformed_instances'], 1
        )


if __name__ == '__main__':
    unittest.main()
