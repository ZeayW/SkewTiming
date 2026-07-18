# NUA-Timer / RTL Timing Prediction

This repository contains the experimental code for NUA-Timer, an RTL timing
prediction flow that builds graph/path representations from synthesized RTL
netlists and predicts endpoint timing with graph neural networks, Transformer
path encoders, and baseline models.

## Directory Layout

```text
src/
  scripts/
    run_train_tcad6.sh        # main NUA-Timer training script
    run_parser_train.sh       # generate training dataset
    run_parser_test.sh        # generate test dataset
    run_test_our_tcad.sh      # test NUA-Timer
    run_test_accnn.sh         # test ACCNN baseline
    run_test_gt.sh            # test Graph Transformer baseline
    run_test_rtltimer.sh      # test RTLTimer/XGBoost path baseline
    run_test_masterrtl.sh     # test MasterRTL/XGBoost path baseline

  parser.py                   # raw netlist/timing parser
  parser_helpers.py           # text parsing and dataset-contract validation
  parser_graph_utils.py       # parser-owned DGL graph construction helpers
  train.py                    # main NUA-Timer train/test entry
  train_gt.py                 # graph-transformer baseline entry
  train_path_xgb.py           # MasterRTL-style path baseline
  train_path_xgb_multi.py     # RTLTimer-style path baseline

  model2.py                   # main NUA-Timer model
  dag_ops.py                  # optimized differentiable DAG/scatter operators
  model.py                    # baseline/path model utilities
  gtmodel.py                  # graph-transformer baseline model
  pathformer4.py              # current path Transformer encoder
  pathgformer.py              # path/GNN encoder components
  transformer.py              # sequence Transformer components
  options.py                  # main CLI options
  options_gt.py               # graph-transformer CLI options
  tee.py                      # stdout tee helper
  utils/                      # legacy shared utilities plus analysis/plot scripts
  abandon/                    # legacy code and old scripts kept for reference
```

The parser uses the tracked `parser_helpers.py` and `parser_graph_utils.py`
modules and does not depend on the ignored `src/utils/` directory.

## Setup

The experiments are normally run on the remote server with the existing conda
environment:

```bash
conda activate dgl_new
```

The configuration currently validated on 149-2 is:

```text
Python 3.10.12
PyTorch 2.1.0
DGL 2.4.0+cu118
NumPy 1.26.4
pandas 2.2.3
scikit-learn 1.7.2
NetworkX 3.4.2
torchmetrics 1.6.1
```

For the closest reproduction, clone or export the server's `dgl_new`
environment. If a fresh environment is needed, start with Python 3.10:

```bash
conda create -n dgl_new python=3.10
conda activate dgl_new

pip install \
  joblib \
  networkx \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  tqdm \
  torchmetrics \
  xgboost \
  matplotlib \
  seaborn \
  adjustText \
  openpyxl
```

Install PyTorch and DGL from their CUDA-specific distributions before the
remaining packages. Match the CUDA runtime on the target machine; the
validated server environment uses the CUDA 11.8 DGL build shown above.

## Running Experiments

Run all scripts from `src/`. The scripts live in `src/scripts/`, but they are
launched from `src/` so their relative dataset/checkpoint paths remain valid:

```bash
cd src
conda activate dgl_new
```

Generate datasets:

```bash
bash scripts/run_parser_train.sh
bash scripts/run_parser_test.sh
```

Parser traversal and split ordering are deterministic by default. Use
`--parser_seed N` to select a different reproducible ordering.
The parser scripts use `--parser_workers 4` to parse independent timing cases
in parallel. Set it to `1` for the serial fallback or when CPU/RAM is limited.
Constant propagation defaults to the event-driven
`--parser_constant_impl worklist` implementation. Use `scan` for the original
full-graph scan or `compare` to execute both implementations and assert exact
agreement while debugging parser changes.

For each design, the parser defines the PO set as the union of endpoints that
have a label in at least one NUIAT case. A case that does not report one of
those endpoints stores `-1` at the corresponding label position. NUA-Timer and
the graph-transformer baseline exclude these rows from supervised losses,
path supervision, metrics, metadata, and saved predictions; zero remains a
valid timing label and is not filtered.

Train the current NUA-Timer TCAD configuration:

```bash
bash scripts/run_train_tcad6.sh
```

This canonical entry defaults to the best validated optimization configuration:

```text
--batch_size 16
--num_epoch 100
--flag_alternate
--lr_scheduler --lr_scheduler_patience 3 --lr_scheduler_factor 0.25
--min_learning_rate 1e-5
--ema_decay 0.999 --ema_start_epoch 5 --ema_scheduler_source raw
```

Override run-specific values without editing the script, for example:

```bash
GPU=1 CHECKPOINT=experiments/my_run NUM_EPOCH=100 BATCH_SIZE=16 \
  bash scripts/run_train_tcad6.sh
```

The main BPN implementation is fixed to this TCAD6 configuration. Deprecated
ablation branches in the model path have been removed; changing those model
mechanism flags in `train.py` will raise an explicit configuration error.
Training evaluates validation and test sets every epoch by default. Add
`--eval_every N` to the training command to run evaluation every `N` epochs
without changing the training updates.
For short smoke/profiling runs, `--max_train_batches N` limits the number of
training batches per epoch and `--debug_case_limit N` limits the number of
timing cases per design. Leave both unset for full experiments. Use
`--po_batch_size N` only for profiling PO batching; the default keeps the
dynamic TCAD6 setting with a memory guard. The guard caps roughly
`num_nodes_in_graph_batch * po_batch_size` through
`--po_batch_node_budget` to avoid OOM when increasing graph `--batch_size`;
set `--po_batch_node_budget 0` only when reproducing the old unconstrained
PO batching behavior.
The canonical runtime implementations are:

```text
--cpe_impl frontier
--mtde_backward_impl custom
--mtde_forward_cache cache
--mtde_forward_impl scatter
--fse_gnn_impl builtin
--fse_eval_cache cache
```

`frontier` keeps the CPE path semantics while avoiding the dense critical-path
mask. The custom MTDE backward uses a compact autograd function, and the MTDE
forward uses cached topology tensors plus segmented scatter operations. The FSE
structure GNN uses DGL built-in aggregation instead of Python message/reduce
UDFs.

Use the corresponding `compare` modes only on small smoke runs: they execute
both implementations and check outputs, attention/correlation values, and the
relevant gradients. The original fallback modes remain available as
`--mtde_backward_impl dgl`, `--mtde_forward_impl dgl`, and
`--fse_gnn_impl udf`.

`--fse_eval_cache cache` is active only under `model.eval()` and
`torch.no_grad()`. It computes the case-invariant FSE node embeddings once per
batched graph and reuses them across timing cases. Training never uses this
cache, and a new cache is built for each later evaluation epoch after model
parameters have changed. Use `--fse_eval_cache compare` for a small evaluation
smoke test or `--fse_eval_cache off` to disable it.

Evaluate NUA-Timer:

```bash
bash scripts/run_test_our_tcad.sh
```

Evaluate baselines:

```bash
bash scripts/run_test_accnn.sh
bash scripts/run_test_gt.sh
bash scripts/run_test_rtltimer.sh
bash scripts/run_test_masterrtl.sh
```

Runtime/statistics logging is disabled by default. Add `--log_level 1` to a
parser, train, or test command only when you want parser runtime logs, test
runtime breakdowns, and `corr_sim` reporting.

CUDA kernel blocking is also disabled by default. Add `--cuda_blocking` only
when debugging asynchronous CUDA errors; it can make normal GPU training much
slower.

The scripts use relative dataset and checkpoint paths. On the remote server,
verify that paths such as `../datasets/...`, `../../datasets/...`, and
`../checkpoints/...` point to the intended experiment directories before
launching a long run.

## Local Editing, Remote Running

Recommended workflow:

1. Edit code locally in `RTL-Timing-Prediction`.
2. Sync only code and scripts to the server; do not sync `checkpoints`,
   `datasets`, `figs`, or other experiment outputs.
3. Run experiments on the server under `conda activate dgl_new`.
4. Keep new one-off experiments in `src/abandon/old_scripts` until they become
   the latest canonical script.

## Cleanup Policy

The `src/` root is reserved for active training, testing, parsing, and model
code. Files not used by the canonical run scripts were moved into:

```text
src/abandon/old_scripts/   # older run scripts superseded by current scripts
src/abandon/legacy_code/   # backup/experimental modules not in active imports
src/utils/                 # plotting, statistics, result collection utilities
```

When adding a new experiment, prefer updating an existing canonical script or
adding one clearly named script. Avoid keeping multiple stale variants for the
same model in `src/`.
