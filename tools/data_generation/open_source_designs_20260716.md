# Open-Source Design Intake (2026-07-16)

This note records open-source RTL collected for the NUA-Timer large-design
test extension. It separates source acquisition from TangDynasty (TD)
compatibility: a downloaded design is not considered dataset-ready until a
one-case frontend and STA smoke test succeeds without unintended black boxes.

## Server Location

All source snapshots are stored on the linux10/HPC shared filesystem under:

```text
/research/d4/gds/ziyiwang21/Code/nuatimer_rawdata/open_source_designs_20260716/sources
```

Pinned sources:

| Source | Revision | Local directory |
|---|---|---|
| HighTide | `e22f73115477c5b9bd9d60f3e4b7669201202519` | `hightide/` |
| OpenROAD-flow-scripts | `bea7dcd7be7f26d1328f6058b01cf42bf4352aa2` | `orfs-bea7dcd7/` |
| VTR/Koios | `c3ad1ec64818c9db312e24967de9d6283410ab19` | `koios-c3ad1ec64818c9db312e24967de9d6283410ab19/` |
| NVDLA nv_small | `771f20cc9e69759d7277978eb41e8d47f1547374` | `nvdla-nv_small/` |
| NVDLA nvdlav1 | `8e06b1b9d85aab65b40d43d08eec5ea4681ff715` | `nvdla-hw/` |
| mor1kx | `f46074a88576e1d7e2fc6cfae14a664dc593a2d8` | `mor1kx/` |
| LiteX | `b6c608fa3343e40f4169e99565c9f443aa3134da` | `litex/` |
| VexRiscv | `680756065e9e6fc50d8c3d6c58191a16e867d822` | `vexriscv/` |

## Recommended Intake Order

Published/open-flow cell counts below are useful only for triage. They are not
directly comparable to NUA-Timer parser node counts and must be replaced by TD
frontend audit results after each smoke test.

| Priority | Design | Domain and top | Collected RTL | Scale evidence | TD intake status |
|---|---|---|---|---|---|
| 1 | Gemmini mesh | Systolic ML accelerator, `gemmini` | HighTide single-file Verilog, 3.67 MB, 7 modules | 376,009 Nangate45 logic cells | Best first smoke candidate. No external SRAM macro references found in the release RTL. |
| 2 | clstm_like.large | Complex LSTM accelerator, `C_LSTM_datapath` | Koios single-file Verilog, 0.87 MB, 81 modules | 1,214,757 VTR cells; 139,379 synthesis nodes | Pure Verilog with behavioral RAM definitions. Strong large, non-CPU candidate. |
| 3 | SweRV EH1 | RISC-V core, `swerv` | ORFS single-file sv2v Verilog, 6.34 MB, 133 modules | ORFS synthesis-qualified | Use bare `swerv`, not `swerv_wrapper`, to avoid platform FakeRAM macros. |
| 4 | NyuziProcessor | Vector/GPGPU processor, `NyuziProcessor` | HighTide sv2v release, 3 RTL files | 377,350 Nangate45 logic cells, 55 SRAM macros | Replace inner FakeRAM black boxes with behavioral SRAM models before TD. |
| 5 | CoralNPU | Google Coral NPU, `CoreMiniAxi` | HighTide release, 1.12 MB plus wrappers | 140,663 Nangate45 logic cells, 2 SRAM macros | Requires two behavioral SRAM replacements; hierarchical synthesis was needed in HighTide. |
| 6 | NVDLA partitions | DNN accelerator partitions | HighTide/NVDLA nv_small, 18.3 MB Verilog | `partition_c`: 250,898; `partition_o`: 189,268 Nangate45 cells | Prefer partitions over full NVDLA. Requires C-preprocessing/file-list handling and RAM models. |
| 7 | Eyeriss v2 | Sparse CNN accelerator, `TOP` | HighTide single-file release, 1.22 MB | 307,701 Nangate45 logic cells, about 193 SRAM macros | Valuable but RAM-model work is large. Run after macro-light designs. |
| 8 | CNN/NNgen | CNN accelerator, `cnn` | HighTide 2.79 MB plus four RAM models | Roughly 200k-400k mapped cells depending on platform/run | 65 SRAM instances and simulation-oriented RAM code need sanitizing. |
| 9 | Ariane/CVA6 snapshot | RV64 core, `ariane` | ORFS single-file sv2v Verilog, 9.51 MB, 66 modules | ORFS synthesis-qualified | Needs behavioral replacements for cache RAM wrappers; lower distribution value after Rocket/BOOM. |

## Additional Collected Designs

- Koios: `lstm`, `tpu_like.large.{os,ws}`, `gemm_layer`, `attention_layer`,
  `spmv`, `dnnweaver`, `bwave_like.float.large`, and the rest of Koios 2.0.
  These add operator diversity, but most are smaller than the first-priority
  candidates in published synthesis results.
- LiteEth and LitePCIe: useful networking/IO diversity, but the mapped standard
  cell counts are much smaller and vendor PHY/SRAM stubs increase integration
  work. They are reserve designs rather than reviewer-facing large cases.
- mor1kx: straightforward multi-file Verilog and a useful fallback CPU, but is
  expected to be smaller than SweRV/Ariane.
- LiteX/VexRiscv-SMP: a configurable design generator, not yet a frozen design.
  Before use, fix the core count, cache sizes, FPU/coherence options, generator
  revisions, and generated Verilog checksum.
- OpenPiton and current CVA6/PULP sources were surveyed but not selected for the
  two-week critical path because their dependency and SystemVerilog integration
  risk is substantially higher than the release-Verilog alternatives above.

## License Notes

- HighTide infrastructure is BSD-3-Clause; each imported design retains its
  upstream license. Gemmini uses a BSD-style license, CNN/CoralNPU/Nyuzi use
  Apache-2.0, and Eyeriss v2 is MIT upstream.
- Koios is distributed under the VTR license.
- SweRV is Apache-2.0; Ariane uses Solderpad Hardware License 0.51.
- NVDLA uses the NVIDIA Open NVDLA License and Agreement v1.0.
- mor1kx uses CERN-OHL-W-2.0; LiteX is BSD-2-Clause; VexRiscv is MIT.

Retain the source snapshot and its license alongside every prepared RTL input.

## Smoke-Test Gate

For each design, run these gates in order:

1. Freeze a single top and combine/preprocess its RTL without changing logic.
2. Run one TD frontend case (`GENERATE_ONLY=1`) and reject unresolved functional
   modules. Memory boundaries may use reviewed behavioral models only.
3. Audit the generated commercial netlist for pre-filter nodes and endpoints.
4. Run one full case with `PATH_NUM=1` and `ENDPOINT_NUM=200000`.
5. Estimate peak RAM and wall time before assigning 10-case and 100-case runs.

The first smoke batch should be Gemmini, clstm_like.large, and bare SweRV. It
does not need SRAM compatibility work and therefore gives the fastest evidence
about which new source can enter the 14-day generation schedule.
