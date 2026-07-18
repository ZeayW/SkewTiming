import json
from pathlib import Path
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from commercial_netlist_parser import (  # noqa: E402
    CommercialNetlistError,
    CommercialNetlistParser,
)


class CommercialNetlistParserTests(unittest.TestCase):
    def parse_text(self, text, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        netlist = Path(directory.name) / "commercial.v"
        netlist.write_text(text, encoding="utf-8")
        return CommercialNetlistParser(netlist, **kwargs).parse()

    def test_tangdynasty_top_isolated_from_behavioral_helpers(self):
        result = self.parse_text(
            """
`timescale 1ns / 1ps
module generated_top(a, b, sel, y, z);
  input [1:0] a;
  input [1:0] b;
  input sel;
  output [1:0] y;
  output z;
  wire [1:0] tmp;
  assign tmp = a;
  AL_MUX mux0 (.i0(tmp), .i1(b), .sel(sel), .o(y));
  eq_w2 cmp0 (.i0(a), .i1(b), .o(z));
endmodule

module eq_w2(i0, i1, o);
  input [1:0] i0;
  input [1:0] i1;
  output o;
  reg o;
  always @(*) begin
    o = i0 == i1;
  end
endmodule
""",
            mode="graph",
        )
        self.assertEqual(result.module_name, "generated_top")
        self.assertEqual(result.signals.declared_bits, 10)
        self.assertEqual(result.endpoint_count, 3)
        self.assertEqual(result.assignment_count, 1)
        self.assertEqual(result.instance_count, 2)
        self.assertEqual(result.graph.num_nodes, 10)
        self.assertEqual(result.graph.num_edges, 12)
        self.assertNotIn(
            "procedural_top_level_logic", result.diagnostics.counts
        )

    def test_ansi_ports_parameter_instance_and_alternate_output_name(self):
        result = self.parse_text(
            """
module top(input a, input b, output logic y);
  NAND2_X1 #(.DRIVE(2)) gate0 (.A(a), .B(b), .ZN(y));
endmodule
""",
            mode="graph",
        )
        self.assertEqual(result.input_count, 2)
        self.assertEqual(result.endpoint_count, 1)
        self.assertEqual(result.graph.num_edges, 2)
        y = result.graph.name_to_id["y"]
        self.assertEqual(result.graph.node_types[y], "nand2")

    def test_vector_ternary_tracks_aligned_data_and_shared_select(self):
        result = self.parse_text(
            """
module top(a, b, sel, y);
  input [1:0] a, b;
  input sel;
  output [1:0] y;
  assign y = sel ? a : b;
endmodule
""",
            mode="graph",
        )
        self.assertEqual(result.graph.num_edges, 6)
        self.assertEqual(result.graph.node_types[result.graph.name_to_id["y[1]"]], "mux")
        self.assertEqual(result.graph.node_types[result.graph.name_to_id["y[0]"]], "mux")

    def test_primitive_positional_ports(self):
        result = self.parse_text(
            """
module top(a, b, y);
  input a, b;
  output y;
  and gate0 (y, a, b);
endmodule
""",
            mode="graph",
        )
        self.assertEqual(result.graph.num_edges, 2)
        self.assertEqual(result.graph.node_types[result.graph.name_to_id["y"]], "and")

    def test_sequential_output_is_a_timing_boundary(self):
        result = self.parse_text(
            """
module top(clk, d, q);
  input clk, d;
  output q;
  DFF_X1 state0 (.D(d), .CLK(clk), .Q(q));
endmodule
""",
            mode="graph",
        )
        q = result.graph.name_to_id["q"]
        self.assertEqual(result.graph.num_edges, 0)
        self.assertTrue(result.graph.is_input[q])
        self.assertEqual(result.diagnostics.counts["sequential_boundary"], 1)

    def test_cell_port_json_supports_nonstandard_output_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            netlist = directory / "top.v"
            ports = directory / "ports.json"
            netlist.write_text(
                "module top(a, result);\n"
                "input a; output result;\n"
                "VENDOR_CELL u0 (.DIN(a), .RESULT_PIN(result));\n"
                "endmodule\n",
                encoding="utf-8",
            )
            ports.write_text(
                json.dumps({
                    "VENDOR_CELL": {
                        "DIN": "input", "RESULT_PIN": "output"
                    }
                }),
                encoding="utf-8",
            )
            result = CommercialNetlistParser.from_cell_port_json(
                netlist, ports, mode="graph"
            ).parse()
        self.assertEqual(result.graph.num_edges, 1)

    def test_unknown_named_output_is_an_error_in_strict_mode(self):
        with self.assertRaisesRegex(CommercialNetlistError, "output port"):
            self.parse_text(
                """
module top(a, y);
  input a;
  output y;
  VENDOR_CELL u0 (.PIN_A(a), .RESULT_PIN(y));
endmodule
""",
                mode="stats",
            )

    def test_stats_mode_does_not_materialise_graph(self):
        result = self.parse_text(
            """
module top(input [3:0] a, output [1:0] y);
  assign y = a[1:0];
endmodule
""",
            mode="stats",
        )
        self.assertIsNone(result.graph)
        self.assertEqual(result.signals.declared_bits, 6)
        self.assertEqual(result.input_count, 4)
        self.assertEqual(result.endpoint_count, 2)
        self.assertEqual(result.summary()["pre_filter_nodes"], 6)

    def test_long_non_ansi_port_list_is_streamed_without_repeated_joining(self):
        ports = ",\n".join("unused_{}".format(index) for index in range(5000))
        result = self.parse_text(
            "module top(\n{}\n);\n"
            "input unused_0;\n"
            "output y;\n"
            "assign y = unused_0;\n"
            "endmodule\n".format(ports),
            mode="stats",
        )
        self.assertEqual(result.module_name, "top")
        self.assertEqual(result.signals.declared_bits, 2)


if __name__ == "__main__":
    unittest.main()
