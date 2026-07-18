import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "data_generation"
sys.path.insert(0, str(TOOLS))

from common import (  # noqa: E402
    build_read_checkpoint_tcl,
    build_endpoint_labels,
    extract_first_module,
    iter_timing_paths,
    normalize_input_levels,
    parse_input_ranges,
    parse_timing_paths,
    prepare_sta_netlist,
    rewrite_sta_tcl,
    sha256_td_netlist_content,
    validate_labeled_golden,
    write_labeled_golden,
)
from generate_cases import collect_td_diagnostics  # noqa: E402
from filter_serialized_labels import is_same_register_q_to_d  # noqa: E402
from patch_bp_multi_top import patch_bp_multi_top  # noqa: E402
from package_dataset import main as package_dataset_main  # noqa: E402
from recover_report_dataset import (  # noqa: E402
    LabelAccumulator,
    _merged_inputs,
    _read_partial_case_inputs,
)
from run_sta import (  # noqa: E402
    DEFAULT_ENDPOINT_NUM,
    DEFAULT_PATH_NUM,
    DEFAULT_TD_THREADS,
)
from validate_dataset import main as validate_dataset_main  # noqa: E402


class DataGenerationTests(unittest.TestCase):
    def test_input_range_parsing_and_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            netlist = Path(directory) / "case.v"
            netlist.write_text(
                "module top;\ninput wire [7:0] data, mask;\ninput clk;\nendmodule\n",
                encoding="utf-8",
            )
            ranges = parse_input_ranges(netlist)
            self.assertEqual(ranges, {"data": (7, 0), "mask": (7, 0)})
            expanded = normalize_input_levels([("data", 3), ("clk", 0)], ranges)
            self.assertEqual(expanded[0], ("data[0]", 3))
            self.assertEqual(expanded[7], ("data[7]", 3))
            self.assertEqual(expanded[8], ("clk", 0))

    def test_timing_report_to_label(self):
        report_text = """
Slack : 0
Begin Point         : data[0]
End Point           : out
Logic Level         : 4
Slack : 0
Begin Point         : data[1]
End Point           : out
Logic Level         : ADDER
ADDER foo
ADDER bar
LUT baz
"""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "timing.rpt"
            report.write_text(report_text, encoding="utf-8")
            paths = parse_timing_paths(report)
            self.assertEqual(paths, [("data[0]", "out", 4), ("data[1]", "out", 3)])
            self.assertEqual(list(iter_timing_paths(report)), paths)
            labels = build_endpoint_labels(
                [("data[0]", 0), ("data[1]", 2)], iter_timing_paths(report)
            )
            self.assertEqual(labels, [("out", "data[1]", 5)])

    def test_endpoint_label_uses_numeric_maximum(self):
        labels = build_endpoint_labels(
            [("small", 3), ("large", 10)],
            [("small", "out", 4), ("large", "out", 1)],
        )
        self.assertEqual(labels, [("out", "large", 11)])

    def test_non_nuiat_startpoint_is_rejected(self):
        diagnostics = {}
        with self.assertRaisesRegex(ValueError, "outside the current case NUIAT"):
            build_endpoint_labels(
                [("data", 2)],
                [("data", "mixed", 3), ("state_reg_q", "mixed", 6)],
                diagnostics=diagnostics,
            )

        self.assertEqual(diagnostics["timing_path_count"], 2)
        self.assertEqual(diagnostics["nuiat_startpoint_path_count"], 1)
        self.assertEqual(diagnostics["unexpected_startpoint_path_count"], 1)

    def test_report_recovery_ignores_non_nuiat_startpoint(self):
        accumulator = LabelAccumulator({"data": 2})
        accumulator.update("state_reg_q", "out", 6)
        accumulator.update("data", "out", 3)
        accumulator.update("clk", "clock_only", 99)

        self.assertEqual(accumulator.labels(), [("out", "data", 5)])
        self.assertEqual(accumulator.matched_path_count, 1)
        self.assertEqual(accumulator.unmatched_path_count, 2)

    def test_same_register_path_filter_handles_vector_bits(self):
        report_text = """
Slack : 0
Begin Point         : scalar_reg_q
End Point           : scalar_reg_d
Logic Level         : 0
Slack : 0
Begin Point         : vector_reg_q[31]
End Point           : vector_reg_d[31]
Logic Level         : 0
Slack : 0
Begin Point         : vector_reg_q[31]
End Point           : vector_reg_d[30]
Logic Level         : 2
"""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "timing.rpt"
            report.write_text(report_text, encoding="utf-8")
            self.assertEqual(
                parse_timing_paths(report),
                [("vector_reg_q[31]", "vector_reg_d[30]", 2)],
            )

    def test_serialized_filter_matches_only_same_register_bit(self):
        self.assertTrue(is_same_register_q_to_d("foo_reg_q", "foo_reg_d"))
        self.assertTrue(is_same_register_q_to_d("foo_reg_q[31]", "foo_reg_d[31]"))
        self.assertFalse(is_same_register_q_to_d("foo_reg_q[31]", "foo_reg_d[30]"))
        self.assertFalse(is_same_register_q_to_d("foo_reg_q[31]", "bar_reg_d[31]"))

    def test_label_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden_labeled.txt"
            write_labeled_golden(output, [("a", 1)], [("z", "a", 4)])
            summary = validate_labeled_golden(output)
            self.assertEqual(summary["pi_count"], 1)
            self.assertEqual(summary["endpoint_count"], 1)
            self.assertEqual(summary["critical_input_count"], 1)
            strict_summary = validate_labeled_golden(
                output, require_nuiat_startpoints=True
            )
            self.assertEqual(strict_summary["critical_input_count"], 1)

    def test_strict_label_validation_rejects_non_nuiat_startpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden_labeled.txt"
            output.write_text(
                "// input constrains\n"
                "// pin, level\n"
                "data 1\n"
                "// pin to pin level synthesised\n"
                "// outpin, critical input, max level\n"
                "out, state_reg_q, 4\n",
                encoding="utf-8",
            )
            validate_labeled_golden(output)
            with self.assertRaisesRegex(ValueError, "is not a NUIAT PI"):
                validate_labeled_golden(
                    output, require_nuiat_startpoints=True
                )

    def test_strict_label_validation_requires_critical_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden_labeled.txt"
            output.write_text(
                "// input constrains\n"
                "// pin, level\n"
                "data 1\n"
                "// pin to pin level synthesised\n"
                "// outpin, max level\n"
                "out, 4\n",
                encoding="utf-8",
            )
            validate_labeled_golden(output)
            with self.assertRaisesRegex(ValueError, "has no critical input"):
                validate_labeled_golden(
                    output, require_nuiat_startpoints=True
                )

    def test_tcl_rewrite(self):
        rewritten = rewrite_sta_tcl(
            "proc run_td {} {\n"
            "    import_device device.db\n"
            "    read_verilog -file case.v -top top\n"
            "    write_verilog read.v -implement -nolf\n"
            "    report_timing_summary -path_num 10000000 -ep_num 100000 -file old.rpt\n"
            "    write_verilog gate.v\n"
            "}\n",
            DEFAULT_PATH_NUM,
            DEFAULT_ENDPOINT_NUM,
            ["data[0]", "data[1]"],
        )
        self.assertIn("set nua_startpoint_names {", rewritten)
        self.assertIn("    data[0]", rewritten)
        self.assertIn("    data[1]", rewritten)
        self.assertIn(
            "set nua_startpoints [get_ports -nowarn $nua_startpoint_names]",
            rewritten,
        )
        self.assertIn("[llength $nua_startpoints] != 2", rewritten)
        self.assertIn(
            "report_timing_path -from $nua_startpoints "
            "-path_num 1 -ep_num 200000 -file timing.rpt",
            rewritten,
        )
        self.assertNotIn("report_timing_summary", rewritten)
        self.assertIn("set_param flow qor_monitor off", rewritten)
        self.assertIn("set_param flow thread {}".format(DEFAULT_TD_THREADS), rewritten)
        self.assertNotIn("write_verilog", rewritten)

    def test_tcl_rewrite_imports_checkpoint(self):
        rewritten = rewrite_sta_tcl(
            "proc run_td {} {\n"
            "    read_verilog -file case.v -top top\n"
            "    report_timing_summary -path_num 10 -ep_num 20 -file old.rpt\n"
            "}\n",
            1,
            200000,
            ["data"],
            checkpoint=Path("/tmp/read.db"),
        )
        self.assertIn(
            "import_db {{{}}}".format(Path("/tmp/read.db").resolve()), rewritten
        )
        self.assertNotIn("read_verilog", rewritten)

    def test_tcl_rewrite_rejects_unsafe_or_duplicate_startpoints(self):
        text = (
            "proc run_td {} {\n"
            "    read_verilog -file case.v -top top\n"
            "    report_timing_summary -path_num 1 -ep_num 10 -file old.rpt\n"
            "}\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rewrite_sta_tcl(text, 1, 10, ["data", "data"])
        with self.assertRaisesRegex(ValueError, "Tcl-unsafe"):
            rewrite_sta_tcl(text, 1, 10, ["bad port"])

    def test_build_read_checkpoint_tcl(self):
        text = (
            "proc run_td {} {\n"
            "    import_device device.db\n"
            "    read_verilog -file case.v -top top\n"
            "    read_sdc case.sdc\n"
            "    optimize_rtl\n"
            "}\n"
            "set state(1_td) [run_td]\n"
        )
        checkpoint_tcl = build_read_checkpoint_tcl(
            text, Path("/tmp/design.v"), Path("/tmp/read.db"), td_threads="8"
        )
        self.assertIn(
            "read_verilog -file {{{}}} -top top".format(
                Path("/tmp/design.v").resolve()
            ),
            checkpoint_tcl,
        )
        self.assertIn("set_param flow thread 8", checkpoint_tcl)
        self.assertIn(
            "export_db {{{}}}".format(Path("/tmp/read.db").resolve()),
            checkpoint_tcl,
        )
        self.assertNotIn("read_sdc", checkpoint_tcl)
        self.assertNotIn("optimize_rtl", checkpoint_tcl)

    def test_td_netlist_hash_ignores_generated_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.v"
            second = Path(directory) / "second.v"
            body = "`timescale 1ns / 1ps\nmodule top; endmodule\n"
            first.write_text(
                "// Verilog netlist created by Tang Dynasty v6.2\n"
                "// Mon Jan 1 00:00:00 2025\n" + body,
                encoding="utf-8",
            )
            second.write_text(
                "// Verilog netlist created by Tang Dynasty v6.2\n"
                "// Tue Jan 2 00:00:00 2025\n" + body,
                encoding="utf-8",
            )
            self.assertEqual(
                sha256_td_netlist_content(first), sha256_td_netlist_content(second)
            )

    def test_extract_first_module(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.v"
            destination = Path(directory) / "target.v"
            source.write_text(
                "// header\nmodule top;\n// synthesis translate_off\ninitial $stop;\n"
                "// synthesis translate_on\nendmodule\nmodule helper; endmodule\n",
                encoding="utf-8",
            )
            extract_first_module(source, destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("module top", text)
            self.assertNotIn("$stop", text)
            self.assertNotIn("module helper", text)

    def test_union_endpoint_dataset_packaging_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projects = root / "projects"
            for index, endpoint in ((0, "out0"), (1, "out1")):
                case = projects / "top_{}".format(index)
                case.mkdir(parents=True)
                (case / "top_{}.v".format(index)).write_text(
                    "module top(input a, output out0, output out1);\n"
                    "assign out0 = a; assign out1 = a; endmodule\n",
                    encoding="utf-8",
                )
                write_labeled_golden(
                    case / "golden_labeled.txt",
                    [("a", index)],
                    [(endpoint, "a", index + 1)],
                )

            strict_output = root / "strict"
            with mock.patch(
                "sys.argv",
                [
                    "package_dataset.py",
                    "--projects-dir",
                    str(projects),
                    "--output-dir",
                    str(strict_output),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "endpoint set differs"):
                    package_dataset_main()

            union_output = root / "union"
            with mock.patch(
                "sys.argv",
                [
                    "package_dataset.py",
                    "--projects-dir",
                    str(projects),
                    "--output-dir",
                    str(union_output),
                    "--endpoint-mode",
                    "union",
                ],
            ):
                self.assertEqual(package_dataset_main(), 0)

            with mock.patch(
                "sys.argv",
                [
                    "validate_dataset.py",
                    "--dataset-dir",
                    str(union_output),
                    "--expected-cases",
                    "2",
                    "--endpoint-mode",
                    "union",
                ],
            ):
                self.assertEqual(validate_dataset_main(), 0)

            manifest = __import__("json").loads(
                (union_output / "manifest.json").read_text(encoding="utf-8")
            )
            summary = manifest["designs"]["top"]
            self.assertEqual(summary["endpoint_mode"], "union")
            self.assertEqual(summary["endpoint_union_count"], 2)
            self.assertEqual(summary["common_endpoint_count"], 0)
            self.assertEqual(summary["endpoint_variant_count"], 2)

    def test_prepare_sta_netlist(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.v"
            destination = Path(directory) / "target.v"
            source.write_text(
                "module top; helper h(); endmodule\n"
                "// synthesis translate_off\ninitial $stop;\n"
                "// synthesis translate_on\n"
                "module helper; endmodule\n"
                "module read_port_1_32_1_0_31_0(input clk); endmodule\n",
                encoding="utf-8",
            )
            status = prepare_sta_netlist(source, destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("module helper", text)
            self.assertNotIn("$stop", text)
            self.assertIn("parameter INIT_VALUE = 0", text)
            self.assertIn('parameter RAM_INIT_STATE = "0"', text)
            self.assertEqual(status["read_port_modules_patched"], 1)
            self.assertEqual(status["translate_off_lines_removed"], 3)

    def test_td_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            (work_dir / "td_1.log").write_text(
                "ok\n"
                "HDL-5340 WARNING: severity system task 'fatal' is ignored\n"
                "HDL-8007 ERROR: missing module\n"
                "HDL-8008 WARNING: unresolved cell is a black box\n",
                encoding="utf-8",
            )
            diagnostics = collect_td_diagnostics(work_dir)
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(any("missing module" in item for item in diagnostics))
            self.assertTrue(any("black box" in item for item in diagnostics))
            self.assertFalse(any("system task 'fatal'" in item for item in diagnostics))
            allowed = collect_td_diagnostics(work_dir, allow_black_boxes=True)
            self.assertEqual(len(allowed), 1)
            self.assertIn("missing module", allowed[0])

    def test_bp_multi_top_compatibility_patch(self):
        source = (
            "module top;\n"
            "  reg [25:0] lce_resp_o;\n"
            "  assign lce_resp_o[23] = 1'b1;\n"
            "  always @(posedge clk_i) begin\n"
            "    if(1'b1) begin\n"
            "      { lce_resp_o[25:25] } <= { 1'b0 };\n"
            "    end\n"
            "  end\n"
            "endmodule\n"
            "module other;\n"
            "  wire [25:0] lce_resp_o;\n"
            "  assign lce_resp_o[23] = 1'b1;\n"
            "endmodule\n"
            "module icache;\n"
            "  reg [95:0] icache_pc_gen_data_o;\n"
            "  assign icache_pc_gen_data_o[95:64] = high;\n"
            "  always @(posedge clk_i) begin\n"
            "      { icache_pc_gen_data_o[63:0] } <= { eaddr_tl_r[63:0] };\n"
            "  end\n"
            "endmodule\n"
        )
        patched = patch_bp_multi_top(source)
        self.assertIn("wire [25:0] lce_resp_o", patched)
        self.assertIn("assign lce_resp_o[25] = 1'b0", patched)
        self.assertNotIn("lce_resp_o[25:25]", patched)
        self.assertIn("wire [95:0] icache_pc_gen_data_o", patched)
        self.assertIn("reg [63:0] icache_pc_gen_data_low", patched)
        self.assertIn(
            "assign icache_pc_gen_data_o[63:0] = icache_pc_gen_data_low",
            patched,
        )

        with self.assertRaises(ValueError):
            patch_bp_multi_top("module top; endmodule\n")

    def test_reference_inputs_only_supply_missing_pin_names(self):
        merged = _merged_inputs([("a", 9), ("b", 8)], [("a", 3), ("c", 4)])
        self.assertEqual(merged, [("a", 3), ("b", 0), ("c", 4)])

    def test_interrupted_case_input_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            golden = Path(directory) / "golden.txt"
            golden.write_bytes(b"// header\na 3\nbroken\x00\x00\n// pin to pin level synthesised\n")
            inputs, skipped = _read_partial_case_inputs(golden)
            self.assertEqual(inputs, [("a", 3)])
            self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
