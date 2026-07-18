# AGENTS.md

This file records the working conventions for agents/collaborators on the
`RTL-Timing-Prediction` codebase.

## Scope

These instructions apply to the entire `RTL-Timing-Prediction` repository.
The LaTeX paper lives in the sibling `NUA-Timer-tex` directory and is not part
of this repository's experiment workflow.

## Basic Workflow

- Edit code locally in:

  ```bash
  /Users/zeayw/Documents/GitHub/NUA-Timer/RTL-Timing-Prediction
  ```

- Run experiments on the remote server, not on the local machine.
- Generate datasets on `hpc2`, including running `src/parser.py` under the
  `dgl_new` conda environment and producing the serialized `.pkl` files. Do
  not run parser/data generation on linux10 or the experiment servers.
- After parser output has been validated on `hpc2`, transfer the completed
  serialized dataset to whichever execution server will run the experiment
  (`149`, `169`, or `170`). Those servers are for training and testing against
  the transferred dataset.
- The primary remote experiment copy is:

  ```bash
  /data/zywang/Codes/RTL-Timing-Prediction
  ```

- Use this conda environment for training and testing on 149/169/170:

  ```bash
  conda activate dgl_new
  ```

- Run experiment commands from the remote `src/` directory so the relative
  paths in scripts remain valid:

  ```bash
  cd /data/zywang/Codes/RTL-Timing-Prediction/src
  ```

## Remote Access

The known working route to the 149-2 experiment server is through `gateway`:

```bash
ssh gateway 'ssh -p 2349 zywang@projgw'
```

For non-interactive checks, use batch mode and the temporary known-hosts file:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 gateway \
  'ssh -o BatchMode=yes \
       -o StrictHostKeyChecking=no \
       -o UserKnownHostsFile=/tmp/nua_timer_known_hosts \
       -o ConnectTimeout=10 \
       -p 2349 zywang@projgw "hostname; pwd"'
```

If the gateway times out, retry once or twice before assuming the remote
machine is down.

## Dataset Generation And Transfer

The required data flow is:

```text
raw netlist/timing data on the shared linux10/HPC filesystem
  -> run src/parser.py on hpc2 under dgl_new
  -> validate data.pkl, split.pkl, and ntype2id.pkl on hpc2
  -> transfer the completed dataset directory to 149/169/170
  -> train or test on the selected execution server
```

The parser execution configuration is:

```bash
ssh hpc2
source /research/d4/gds/ziyiwang21/miniconda3/etc/profile.d/conda.sh
conda activate dgl_new
cd /research/d4/gds/ziyiwang21/Code/RTL-Timing-Prediction/src
bash scripts/run_parser_train.sh
# Or, for test data:
bash scripts/run_parser_test.sh
```

The shared `dgl_new` parser environment currently uses Python 3.10.12,
PyTorch 2.1.0, and DGL 2.4.0. Its CUDA 12.1 build differs from the CUDA 11.8
build on 149, but parser execution is CPU-based and has been validated against
the same parser/DGL tests. Do not use the legacy `dgl_3090` environment for new
parser runs.

Keep raw data, parser work directories, and incomplete parser output on the
shared linux10/HPC filesystem, but run the parser process on `hpc2` only.
Transfer only a completed, validated serialized dataset to the execution
servers. Do not generate `.pkl` datasets independently on multiple machines,
because that can introduce parser-version or split drift.

## Sync Policy

The local checkout is the only source-editing workspace. Make every source,
script, configuration, documentation, and test change locally. Never hot-fix
repository files on 149, 169, 170, linux10, or an HPC node, including with an
editor, `sed`, `cp`, or `rsync`.

After every completed code change:

1. Validate the change locally and inspect `git status --short` and
   `git diff --stat`.
2. Commit the intended files locally. Do not include unrelated user changes.
3. Immediately push the commit to GitHub with `git push origin main`.
4. Pull that commit on 149, 169, 170, and linux10 with
   `git pull --ff-only origin main`.
5. Verify that every server reports the same commit as local with
   `git rev-parse HEAD`. Do not start an experiment with a stale checkout.

The experiment-server repository is:

```text
/data/zywang/Codes/RTL-Timing-Prediction
```

The linux10 repository is:

```text
/research/d4/gds/ziyiwang21/Code/RTL-Timing-Prediction
```

HPC1-HPC8 share the linux10 project filesystem, so pulling once in the linux10
repository updates the source visible to those nodes; do not create independent
HPC source copies.

Use Git, not `rsync`, to distribute repository source. `rsync` remains
appropriate for datasets and other generated artifacts that are intentionally
excluded from Git. If a server pull is blocked by tracked modifications or an
untracked path, do not reset, stash, delete, or overwrite it. Inspect the
difference, preserve any unique work, reproduce the required change locally,
commit and push it, and only then make the server checkout match GitHub.

Never commit or copy experiment outputs back into source control. In
particular, preserve and keep out of Git:

- `checkpoints/`
- `datasets/`
- `figs/`
- `rawdata/`
- logs, spreadsheets, archives, and other generated analysis artifacts

## Code Layout

- `src/train.py`: main NUA-Timer train/test entry.
- `src/model2.py`: main NUA-Timer model implementation.
- `src/parser.py`: dataset generation from raw netlist/timing data.
- `src/scripts/`: canonical run scripts.
- `src/utils/`: plotting, result collection, runtime/statistics utilities.
- `src/abandon/`: old scripts and legacy experiments kept for reference only.

Keep active training, testing, parsing, and model code in `src/`. Do not bring
back abandoned variants unless the user explicitly asks.

## Canonical TCAD6 Training Configuration

The current main training script is:

```bash
src/scripts/run_train_tcad6.sh
```

Launch it from `src/`:

```bash
cd /data/zywang/Codes/RTL-Timing-Prediction/src
source /data/zywang/anaconda3/etc/profile.d/conda.sh
conda activate dgl_new
bash scripts/run_train_tcad6.sh
```

The canonical model/training parameters are defined by this script. Current
important fixed settings include:

```text
--quick
--hidden_dim 128
--batch_size 16
--num_epoch 100
--flag_width
--flag_reverse
--flag_path_supervise
--flag_alternate
--global_info_choice 12
--global_cat_choice 10
--flag_transformer 1
--use_corr_pe
--use_attn_bias
--base_pe 2
--flag_filter
--path_feat_choice 3
--path_corr_choice 1
--flag_rawpath
--use_pathgnn
--alpha 5
--beta 5
--path_delay_choice 3
--flag_residual
--lr_scheduler
--lr_scheduler_patience 3
--lr_scheduler_factor 0.25
--min_learning_rate 1e-5
--ema_decay 0.999
--ema_start_epoch 5
--ema_scheduler_source raw
--gpu 0
```

Do not change the mechanism flags casually. The refactored code assumes this
TCAD6 configuration for the main BPN path.

## GPU Selection

Before launching a run, check GPU availability:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader
```

Prefer an idle RTX 3090. Avoid using the Quadro P620 for long training runs.
Select the GPU with the command-line `--gpu` argument. If a run requires a
script change, make it locally, commit and push it, and pull it on every server
before launching the run. Do not temporarily edit a server-side script.

## Safe Remote Launch Pattern

Use `nohup` or a small temporary shell script for long remote runs. If the
default checkpoint directory already exists, create a timestamped checkpoint
path instead of deleting or overwriting it.

Example pattern:

```bash
cd /data/zywang/Codes/RTL-Timing-Prediction/src
source /data/zywang/anaconda3/etc/profile.d/conda.sh
conda activate dgl_new

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG=/data/zywang/Codes/RTL-Timing-Prediction/train_tcad6_${RUN_ID}.log

nohup bash scripts/run_train_tcad6.sh > "$LOG" 2>&1 &
echo "TRAIN_STARTED PID=$! LOG=$LOG"
```

Note: `train.py` may redirect stdout/stderr again into the selected checkpoint
directory as `stdout.log`, `stderr.log`, and `res.txt`. If the outer log is
empty, inspect the checkpoint logs before assuming the run failed.

## Runtime and Statistics Logging

Runtime/statistics reporting is disabled by default. Enable it only when
needed with:

```bash
--log_level 1
```

This controls extra runtime/statistics work such as parser timing logs, test
runtime breakdowns, and `corr_sim` reporting.

## Standard Experiment Scripts

Run the parser scripts from `src/` on `hpc2` under `dgl_new`:

```bash
bash scripts/run_parser_train.sh
bash scripts/run_parser_test.sh
```

Run the training and test scripts from `src/` on 149/169/170 under `dgl_new`:

```bash
bash scripts/run_train_tcad6.sh
bash scripts/run_test_our_tcad.sh
bash scripts/run_test_accnn.sh
bash scripts/run_test_gt.sh
bash scripts/run_test_rtltimer.sh
bash scripts/run_test_masterrtl.sh
```

## Git Hygiene

- Keep generated files out of git.
- Do not commit `src/abandon/`, `src/scripts/`, or `src/utils/` unless the user
  explicitly changes that policy.
- Do not commit `.DS_Store`, figures, checkpoints, datasets, spreadsheets, zip
  files, or runtime logs.
- Before committing or syncing, inspect:

  ```bash
  git status --short
  git diff --stat
  ```

## Validation

For code edits, prefer lightweight local syntax/import checks when possible.
Run parser validation on `hpc2` under `dgl_new`, and run the real
training/testing workload on 149/169/170 under `dgl_new`.

Useful local checks:

```bash
python -m py_compile src/train.py src/model2.py src/options.py
```

Remote environment sanity check:

```bash
source /data/zywang/anaconda3/etc/profile.d/conda.sh
conda activate dgl_new
python - <<'PY'
import torch
import dgl
print("torch", torch.__version__)
print("dgl", dgl.__version__)
print("cuda", torch.cuda.is_available())
PY
```
