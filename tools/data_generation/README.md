# NUA-Timer Raw-Data Generation

This directory contains the reproducible wrapper around the TangDynasty data
generation flow. The commercial binary, generated netlists, timing reports,
and datasets are not stored in Git.

## Pipeline

Run the following commands on `linux10-2`. No conda environment is required
for TangDynasty; the wrapper configures its own libraries and license path.

```bash
TD=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/TD_6.2.0_Release_152540
TOOLS=/research/d4/gds/ziyiwang21/Code/nuatimer_data_generation
WORK=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/work/lenet_smoke

python3 "$TOOLS/generate_cases.py" \
  --td-root "$TD" \
  --rtl /path/to/lenet.v \
  --design-name lenet \
  --work-dir "$WORK" \
  --sdcs 1 \
  --execute

python3 "$TOOLS/run_sta.py" \
  --td-root "$TD" \
  --projects-dir "$WORK/projects" \
  --workers 1

python3 "$TOOLS/package_dataset.py" \
  --projects-dir "$WORK/projects" \
  --output-dir /research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/cases_round7_v6_large_smoke

python3 "$TOOLS/validate_dataset.py" \
  --dataset-dir /research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/cases_round7_v6_large_smoke \
  --expected-cases 1
```

Packaging defaults to `--endpoint-mode strict`, which requires every case to
contain the same endpoint set. Test-only `path_num=1` datasets may legitimately
have case-specific endpoint coverage. Package and validate those datasets with
`--endpoint-mode union`; labels remain case-specific, and `src/parser.py` builds
the endpoint union while storing `-1` for a case/endpoint pair that is absent.

To combine independently generated designs and transfer the validated raw
dataset to the training server, use:

```bash
SOURCE_DATASETS=/path/to/BOOM/dataset:/path/to/RocketCore/dataset \
MERGED_DATASET=/path/to/cases_round7_v9_boom_rocket \
DESTINATION_DIR=/data/zywang/Codes/RTL-Timing-Prediction/rawdata/cases_round7_v9_boom_rocket \
  "$TOOLS/merge_and_transfer_datasets.sh"
```

The destination must be new. Source manifests are retained as named JSON files
at the merged dataset root, and validation runs before transfer.

After the smoke test passes, use a new work directory with `--sdcs 10` for the
pilot and then another new directory with `--sdcs 100` for the final dataset.
Never reuse a non-empty incomplete generation directory: the wrapper refuses
to overwrite it.

`run_sta.py` is resumable. A valid `golden_labeled.txt` is skipped, while a
failed case retains its Tcl and timing report for diagnosis. By default it
disables TD's QoR monitor, omits the `read.v`/`gate.v` debug netlists, and
removes the timing report after its labels have been parsed. Use
`--qor-monitor` or `--keep-intermediates` only when diagnosing a flow problem.

Cases of the same design share one prepared frontend netlist. When at least two
cases are pending, `run_sta.py` runs `read_verilog` once, exports a pre-SDC
`read.db`, and uses `import_db` for every case. It verifies that all case
netlists have identical content after ignoring TD's generated-at timestamp.
Use `--no-read-checkpoint` only for an A/B comparison or checkpoint diagnosis;
timing-driven `optimize_rtl`, `map`, and `pack` remain case-specific.

`--workers` controls independent case-level TD processes. Start with two on an
HPC node, then try four after checking peak memory and available licenses; each
worker can consume the full memory of one design flow. `--td-threads` controls
TD's internal thread setting independently and defaults to `auto`. A current
TD 6.2 A/B run showed no speedup from forcing eight internal threads, so use
case-level workers for throughput. The timing-report defaults are
`--path-num 1` and `--endpoint-num 200000`; they retain the scalar worst-path
label for test-only datasets without enumerating tied critical inputs. Pass a
larger `--path-num` explicitly when generating path-supervision labels.

For each case, `run_sta.py` reads the NUIAT input names from that case's
`*_golden.txt`, expands packed buses to bit-level ports, and injects the exact
set into `report_timing_path -from`. TD therefore searches only paths starting
at inputs that have a case-specific NUIAT value. The generated Tcl verifies
that every requested name resolves to a TD port before starting the report.
Label reduction rejects any unexpected startpoint that is not in the current
case NUIAT table. Each completed `nua_label.status.json` records the requested,
observed, and unexpected startpoint counts for coverage auditing.
The multi-host coordinator treats a case as complete only when the status
confirms that every reported timing path starts at a NUIAT PI. Packaging and
dataset validation independently require every critical input written to the
golden file to appear in that case's NUIAT input table.

## Commercial-Netlist Parser and Structural Audit

`src/commercial_netlist_parser.py` is a separate streaming frontend for large
commercial Verilog netlists. It parses one selected structural module (the first
module by default), so behavioral RAM/mux/helper definitions appended by the
commercial tool are treated as library code rather than top-level graph logic.
It supports ANSI and non-ANSI declarations, vectors and concatenations,
continuous expressions, parameterized instances, named/positional ports,
common vendor output-pin names, and sequential-cell timing boundaries.

To validate and measure a generated netlist before timing labels are available,
use the bounded-memory `stats` mode:

```bash
python3 "$TOOLS_SOURCE/audit_parser_structure.py" \
  --netlist /path/to/design.v \
  --mode stats \
  --output-json /path/to/audit/design.json
```

`stats` validates every statement in the selected module and reports declared
bit nodes, input bits, output endpoint bits, assignments, and instances without
building a Python/DGL graph. Use `--mode graph --reachability` when the compact
bit-level graph and output-fanin filtering are also required. For a non-first
top module, pass `--top-module NAME`.

Unknown vendor output-pin conventions fail closed. Supply a JSON direction map
with `--cell-port-json`, for example:

```json
{
  "VENDOR_CELL": {
    "A": "input",
    "RESULT_PIN": "output"
  }
}
```

`--permissive` exists for diagnostics only: it records unsupported statements
but its counts must not be treated as a validated parse. Every declared output
bit is a structural endpoint candidate, not a replacement for the STA endpoints
in a completed label.

The normal `parser.py` training path still requires labeled golden files and is
unchanged by this audit mode.

Before STA, `run_sta.py` streams the generated netlist through a narrow TD
compatibility pass. It removes `synthesis translate_off` regions and adds the
missing `INIT_VALUE` and `RAM_INIT_STATE` declarations to generated
`read_port_*` helper modules. Other helper modules remain intact. The status
JSON records the full/prepared sizes and number of declarations patched.

## linux10 Scheduler

Do not run TangDynasty directly in the `linux10` login session. That account's
login cgroup is limited to 200 MB and 0.2 CPU, which forces large TD processes
to thrash in swap. Submit `run_slurm_pipeline.sh` to a compute node instead.
The script stages the commercial executable, RTL, and work directory on the
node-local filesystem, then copies only the packaged dataset and audit logs
back to shared storage.

The current account is associated with the `gpu` QoS. A one-case smoke job can
be submitted as follows (the GPU is allocated by the site queue but is not used
by TangDynasty):

The compute-node image lacks three XCB libraries required by the bundled Qt
runtime. They are staged separately under `TOOLS_SOURCE/compat_lib` and copied
into the node-local TD library directory; the commercial installation remains
unchanged. The wrapper also stages TD's `ip/` directory: the sanitized
frontend netlist resolves `add_*`, `eq_*`, and memory-port primitives through
`ip/apm/apm_modules.xml` during STA.

```bash
export TD_SOURCE=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/TD_6.2.0_Release_152540
export TOOLS_SOURCE=/research/d4/gds/ziyiwang21/Code/nuatimer_data_generation
export RTL_SOURCE=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/candidates_20260713/koios/lenet.v
export DESIGN=lenet
export RESULT_ROOT=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/smoke_20260713/lenet
export SDC_COUNT=1

sbatch -p gpu_2h --qos=gpu -A gpu -N 1 -n 1 -c 2 \
  --mem=16G --gres=gpu:1 --time=02:00:00 \
  --job-name=nua_lenet_smoke \
  --output="$RESULT_ROOT/slurm_%j.out" \
  --export=ALL \
  "$TOOLS_SOURCE/run_slurm_pipeline.sh"
```

Set `GENERATE_ONLY=1` in the exported job environment to stop after frontend
case generation. This is useful for screening unsupported RTL and recording
generated netlist sizes before spending commercial STA license time.
`PATH_NUM` and `ENDPOINT_NUM` default to `1` and `200000`; both the Slurm and
direct-HPC launchers pass them through to `run_sta.py`. Override them in the
exported environment when a dataset needs multiple paths per endpoint. The
same launchers accept `STA_WORKERS`, `TD_THREADS`, `READ_CHECKPOINT`, and
`QOR_MONITOR`. `READ_CHECKPOINT=1` and `QOR_MONITOR=0` are the defaults.

Some Chipyard exports intentionally leave SRAM macros and simulation-only
`plusarg_reader` modules external. If TD returns success and produces every
expected case file, set `ALLOW_BLACK_BOXES=1` to reproduce the existing
BOOM/Rocket frontend behavior. This setting is recorded in `job_metadata.txt`
and `generation_status.json`. Keep it disabled for unfamiliar RTL: otherwise a
missing functional module could silently reduce the generated timing graph.

`bp_quad` requires the behavioral SRAM replacements in
`rtl_compat/bp_quad_memory_models.v`. Build its single-file commercial-tool
input with `combine_verilog.py`, placing the memory models before the sv2v
source. The replacements preserve active-low chip/write enables and write-mask
polarity so the macros are inferred as RAMs rather than treated as black boxes.

`bp_multi_top` needs a separate compatibility conversion because its pickled
netlist mixes continuous and procedural assignments on `lce_resp_o`. Run
`patch_bp_multi_top.py` to produce a new input file; the source is never edited
in place. The conversion turns the bit-25 constant register assignment into an
equivalent constant net assignment and checks that the expected source pattern
occurs exactly once.

## Direct HPC Hosts

The shared filesystem is also mounted on `hpc1` through `hpc8`. From
`linux10`, launch an independent pipeline on one host with:

```bash
export HPC_HOST=hpc1
export TD_SOURCE=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/TD_6.2.0_Release_152540
export TOOLS_SOURCE=/research/d4/gds/ziyiwang21/Code/nuatimer_data_generation
export RTL_SOURCE=/path/to/design.v
export DESIGN=my_design
export RESULT_ROOT=/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/hpc_runs/my_design
export SDC_COUNT=1
export STA_WORKERS=2

"$TOOLS_SOURCE/launch_hpc_pipeline.sh"
```

The launcher returns after recording the remote hostname and PID in
`hpc_launcher.txt`. The detached job writes `hpc_pipeline.log`, the same audit
files as the Slurm flow, and a final `audit/exit_code.txt`. Every launch must
use a new result directory. Direct HPC hosts mount `/tmp` as a 10 GB tmpfs. For
large generated netlists, set `SCRATCH_PARENT=/dev/shm`; the current hosts
provide a much larger tmpfs there. Both locations consume system memory, so
budget the generated files together with every concurrent TD worker.

## Output Contract

The packaged dataset follows the layout consumed by `src/parser.py`:

```text
cases_round7_v6_large/
  manifest.json
  design/
    design.v
    design_0/golden.txt
    design_1/golden.txt
```

Every label contains input arrival levels followed by the marker
`// pin to pin level synthesised` and endpoint/critical-input/level tuples.

When combining report-recovered cases and complete labeled variants, preserve
case-specific endpoint coverage with:

```bash
python3 "$TOOLS/merge_labeled_recovery_dataset.py" \
  --projects-dir /path/to/projects \
  --recovered-dir /path/to/recovered \
  --output-dir /path/to/merged_union \
  --design BOOM \
  --design RocketCore \
  --endpoint-mode union
```

In `union` mode each `golden.txt` keeps the labels actually available for that
case. `src/parser.py` builds the graph PO set from their union and serializes a
missing case/endpoint label as `-1`. The default `intersection` mode remains
available for reproducing older packaged datasets.

## Recovering Retained Reports

`recover_report_dataset.py` can recover a small, explicitly provisional dataset
when a generated case still has its complete `timing.rpt`, but the labeled
golden file or zero-delay report is missing. It streams multi-GB reports, uses a
complete reference golden file only to recover the PI name set, and records all
source cases in `recovery_manifest.json`.

The recovered zero-delay labels cover only paths retained in the nonzero-NUIAT
reports. Such datasets are suitable for parser, memory, runtime, and exploratory
inference tests. They must not be reported as fresh STA accuracy results.

For recovered designs with fewer than 100 cases, run `parser.py` with
`--min_cases_per_design 1`. The default remains 100 for normal datasets.
