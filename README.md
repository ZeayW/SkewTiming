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
  train.py                    # main NUA-Timer train/test entry
  train_gt.py                 # graph-transformer baseline entry
  train_path_xgb.py           # MasterRTL-style path baseline
  train_path_xgb_multi.py     # RTLTimer-style path baseline

  model2.py                   # main NUA-Timer model
  model.py                    # baseline/path model utilities
  gtmodel.py                  # graph-transformer baseline model
  pathformer4.py              # current path Transformer encoder
  pathgformer.py              # path/GNN encoder components
  transformer.py              # sequence Transformer components
  options.py                  # main CLI options
  options_gt.py               # graph-transformer CLI options
  tee.py                      # stdout tee helper
  utils/                      # shared core utilities plus analysis/plot scripts
  abandon/                    # legacy code and old scripts kept for reference
```

`src/utils/__init__.py` is the former `src/utils.py`, so existing imports such
as `from utils import *` still work when commands are run from `src/`.

## Setup

The experiments are normally run on the remote server with the existing conda
environment:

```bash
conda activate dgl_new
```

If a fresh environment is needed, use Python 3.7 and install the same major
dependencies as the original setup:

```bash
conda create -n dgl_new python=3.7
conda activate dgl_new

pip install \
  joblib==1.2.0 \
  networkx==2.2 \
  numpy==1.21.5 \
  pandas==1.3.5 \
  scikit-learn==1.0.2 \
  scipy==1.7.3 \
  tqdm==4.64.0 \
  torchmetrics==0.9.3 \
  xgboost \
  matplotlib \
  seaborn \
  adjustText \
  openpyxl
```

Install PyTorch and DGL according to the CUDA version on the server. The
original environment used:

```text
torch==1.13.0
dgl==0.8.1
```

For CUDA 11.3, the old setup used `torch==1.13.0+cu113` and
`dgl_cu113==0.8.1`.

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

Train the current NUA-Timer TCAD configuration:

```bash
bash scripts/run_train_tcad6.sh
```

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
