# BasicSTFM

BasicSTFM is a BasicTS-inspired research framework for spatio-temporal foundation models. It is designed for experiments where the model architecture, pretraining objective, fine-tuning protocol, loss function, metric suite, and training schedule must all remain configurable and extensible.

The framework follows the configuration-centric philosophy of [BasicTS](https://github.com/GestaltCogTeam/BasicTS), while specializing the workflow for foundation-model research: masked spatio-temporal pretraining, forecasting fine-tuning, staged transfer, and custom task design.

## Highlights

- Config-driven experiments with YAML and JSON support.
- Registry-based component discovery for models, datamodules, tasks, losses, metrics, and trainers.
- First-class multi-stage training for pretraining, fine-tuning, and evaluation.
- Custom training flows through task-level batch logic.
- Custom model, loss, metric, and task injection through `custom_imports`.
- Built-in synthetic data for immediate smoke testing without external datasets.
- Built-in `TinySTFoundationModel` and `MLPForecaster` baselines.
- BasicTS-style scale/rescale flow: scale model inputs and compute losses/metrics after inverse transformation.
- Conda-first setup with `requirements.txt` and editable installation.

## Design Overview

BasicSTFM separates an experiment into composable components:

```text
Config       describes the experiment
Registry     resolves component names to Python classes
DataModule   provides train/validation/test dataloaders
Model        computes forward outputs from spatio-temporal tensors
Task         defines the per-batch training logic of one stage
Loss         defines optimization objectives
Metric       defines evaluation signals
Trainer      executes one or more stages
```

This separation is the central design choice. A forecasting fine-tuning stage, a masked reconstruction pretraining stage, and a contrastive learning stage can share the same trainer while using different task implementations.

## Repository Structure

```text
BasicSTFM/
  .gitignore
  README.md
  requirements.txt
  pyproject.toml
  data/
    README.md
    .gitkeep
  configs/
    examples/
      forecasting.yaml
      multistage_pretrain_finetune.yaml
      custom_components.yaml
      file_forecasting.yaml
  examples/
    __init__.py
    custom_model.py
    custom_loss.py
    custom_task.py
  scripts/
    data/
      prepare_npz.py
      inspect_npz.py
  src/
    basicstfm/
      cli.py
      config.py
      registry.py
      builders.py
      data/
      models/
      tasks/
      losses/
      metrics/
      optim/
      engines/
      utils/
  tests/
```

Important files:

- `src/basicstfm/registry.py`: central registry implementation.
- `src/basicstfm/engines/stage.py`: multi-stage training plan parser.
- `src/basicstfm/engines/trainer.py`: default multi-stage trainer.
- `src/basicstfm/models/st_foundation.py`: tiny spatio-temporal foundation model.
- `configs/examples/forecasting.yaml`: single-stage forecasting example.
- `configs/examples/multistage_pretrain_finetune.yaml`: masked pretraining followed by forecasting fine-tuning.
- `configs/examples/custom_components.yaml`: example using user-defined model and loss.
- `configs/examples/file_forecasting.yaml`: generic file-backed forecasting template.
- `data/README.md`: expected dataset layout and preprocessing guide.
- `scripts/data/prepare_npz.py`: generic converter for `.h5`, `.pkl`, `.npy`, `.npz`, `.csv`, and `.txt`.
- `scripts/data/inspect_npz.py`: utility for checking stored array keys and shapes.
- `examples/custom_model.py`: custom model example.
- `examples/custom_loss.py`: custom loss example.
- `examples/custom_task.py`: custom task example.

## Installation From GitHub

The following instructions assume that Conda is already available on your machine. The recommended Python version is **Python 3.10**.

Why Python 3.10:

- BasicSTFM declares `requires-python >= 3.9`.
- The dependency list pins `torch==2.6.0`, and Python 3.10 is a conservative, widely supported choice for PyTorch CPU and CUDA workflows.
- Python 3.10 avoids unnecessary compatibility issues that may appear with very new Python versions on research servers.

### Step 1: Create a Conda Environment

```bash
conda create -n basicstfm python=3.10 -y
conda activate basicstfm
```

Verify the active Python interpreter:

```bash
python --version
which python
```

Expected Python version:

```text
Python 3.10.x
```

Expected interpreter path should contain the Conda environment name:

```text
.../envs/basicstfm/bin/python
```

### Step 2: Clone the Repository From GitHub

Use HTTPS:

```bash
git clone https://github.com/<your-org-or-username>/BasicSTFM.git
cd BasicSTFM
```

Or use SSH:

```bash
git clone git@github.com:<your-org-or-username>/BasicSTFM.git
cd BasicSTFM
```

If `git` is not available on Ubuntu, install it with:

```bash
sudo apt update
sudo apt install -y git
```

If you already cloned the repository and want the latest version:

```bash
cd BasicSTFM
git pull --ff-only
```

Verify the repository content:

```bash
ls
```

Expected files include:

```text
README.md
requirements.txt
pyproject.toml
configs
examples
src
tests
```

### Step 3: Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Then install BasicSTFM in editable mode:

```bash
pip install -e .
```

Editable mode is recommended for research development because changes under `src/basicstfm/` take effect without reinstalling the package.

### Step 5: PyTorch Notes

The default `requirements.txt` pins:

```text
torch==2.6.0
```

This pin is intentional. It avoids accidentally installing a newer PyTorch wheel that may require a newer NVIDIA driver than the one available on a research server.

After installation, check whether PyTorch can see CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

If CUDA is still unavailable, your NVIDIA driver may be too old for the installed PyTorch wheel. In that case, update the driver or install a PyTorch wheel that matches your driver.

To reinstall the pinned dependency set from scratch:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
```

### Step 6: Verify the Command Line Interface

```bash
basicstfm --help
```

Expected output should include:

```text
BasicSTFM experiment runner
```

If the console script is not available, use the module entry point:

```bash
PYTHONPATH=src python -m basicstfm.cli --help
```

## Minimal Verification

Run the lightweight unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Expected output:

```text
Ran 7 tests in ...

OK
```

Run a syntax compilation check:

```bash
PYTHONPYCACHEPREFIX=/tmp/basicstfm_pycache python -m compileall src tests examples
```

Run a dry run to verify the multi-stage parser:

```bash
basicstfm dry-run configs/examples/multistage_pretrain_finetune.yaml
```

Expected output is a JSON-like stage description:

```json
[
  {
    "name": "masked_pretrain",
    "epochs": 5,
    "task": "MaskedReconstructionTask",
    "losses": ["mse"],
    "metrics": ["mae"]
  },
  {
    "name": "forecast_finetune",
    "epochs": 3,
    "task": "ForecastingTask",
    "losses": ["mae", "mse"],
    "metrics": ["mae", "rmse"]
  }
]
```

## Quick Start

### Run a Single-Stage Forecasting Demo

This example uses synthetic data, so no external dataset is required.

```bash
basicstfm train configs/examples/forecasting.yaml
```

For a fast CPU smoke test:

```bash
basicstfm train configs/examples/forecasting.yaml \
  --cfg-options pipeline.stages.0.epochs=1 trainer.device=cpu
```

The run writes outputs to:

```text
runs/synthetic_forecasting/
```

Checkpoints are saved under:

```text
runs/synthetic_forecasting/checkpoints/
```

Typical checkpoint files:

```text
forecasting_best.pt
forecasting_last.pt
```

### Run a Multi-Stage Pretrain-Finetune Demo

This example executes two stages:

1. `masked_pretrain`: randomly masks spatio-temporal observations and reconstructs them.
2. `forecast_finetune`: loads the previous checkpoint and fine-tunes the model for forecasting.

Run the full example:

```bash
basicstfm train configs/examples/multistage_pretrain_finetune.yaml
```

Run a fast CPU smoke test:

```bash
basicstfm train configs/examples/multistage_pretrain_finetune.yaml \
  --cfg-options pipeline.stages.0.epochs=1 pipeline.stages.1.epochs=1 trainer.device=cpu
```

The essential config pattern is:

```yaml
pipeline:
  stages:
    - name: masked_pretrain
      task:
        type: MaskedReconstructionTask
        mask_ratio: 0.35
      losses:
        - type: mse

    - name: forecast_finetune
      load_from: previous
      task:
        type: ForecastingTask
      losses:
        - type: mae
        - type: mse
          weight: 0.05
```

`load_from: previous` means the second stage loads the checkpoint produced by the first stage.

## Command-Line Overrides

Config values can be overridden without editing YAML files.

Run on CPU:

```bash
basicstfm train configs/examples/forecasting.yaml \
  --cfg-options trainer.device=cpu
```

Run one epoch:

```bash
basicstfm train configs/examples/forecasting.yaml \
  --cfg-options pipeline.stages.0.epochs=1
```

Change batch size:

```bash
basicstfm train configs/examples/forecasting.yaml \
  --cfg-options data.batch_size=8
```

Override multiple values:

```bash
basicstfm train configs/examples/forecasting.yaml \
  --cfg-options trainer.device=cpu data.batch_size=8 pipeline.stages.0.epochs=1
```

List indices are addressed by integer positions:

```text
pipeline.stages.0.epochs=1
pipeline.stages.1.optimizer.lr=0.0001
```

## Data Directory and Dataset Preparation

BasicSTFM includes a `data/` directory, but large datasets are ignored by Git. This keeps the repository lightweight while still documenting the expected on-disk layout.

The recommended structure is:

```text
data/
  README.md
  <DATASET_NAME>/
    raw/
      original_files_from_provider
    data.npz
    adj.npz
    README.md
```

Example:

```text
data/
  MY_DATASET/
    raw/
      measurements.h5
      adjacency.pkl
    data.npz
    adj.npz
    README.md
```

The `.gitignore` keeps downloaded data out of version control:

```text
data/**
!data/
!data/.gitkeep
!data/README.md
!data/*/
!data/*/README.md
!data/examples/
!data/examples/**
```

This means Git tracks the data instructions, but not large benchmark files.

### Downloading Data

BasicSTFM does not vendor benchmark datasets. Download datasets from their official source, benchmark repository, or lab storage.

For datasets commonly used with BasicTS-style traffic forecasting experiments, consult the data preparation links in the [BasicTS repository](https://github.com/GestaltCogTeam/BasicTS) and place the downloaded files under `data/<DATASET_NAME>/raw/` before conversion.

Create a dataset directory:

```bash
mkdir -p data/<DATASET_NAME>/raw
```

Download with `wget` if you have a direct URL:

```bash
wget -O data/<DATASET_NAME>/raw/<RAW_FILE_NAME> "<DATASET_URL>"
```

Or copy files from local storage:

```bash
cp /path/to/raw/files/* data/<DATASET_NAME>/raw/
```

### Canonical Data Format

The canonical time-series file is:

```text
data/<DATASET_NAME>/data.npz
```

It should contain key `data`:

```text
data: [T, N, C]
```

Where:

- `T` is the number of timesteps.
- `N` is the number of spatial nodes.
- `C` is the number of channels.

The optional graph file is:

```text
data/<DATASET_NAME>/adj.npz
```

It should contain key `adj`:

```text
adj: [N, N]
```

### Do I Need Preprocessing?

You need preprocessing if your raw files are not already stored as `[T, N, C]`.

Common cases:

- Raw `[T, N]`: add a channel dimension and save as `[T, N, 1]`.
- Raw `[T, N, C]`: save directly as `data.npz` with key `data`.
- Raw CSV matrix `[T, N]`: convert with `--add-channel`.
- Raw HDF5 file: inspect keys with `--list-keys`, then convert with `--input-key`.
- Raw pickle adjacency tuple/list: use `--key adj`; DCRNN/BasicTS-style tuples default to index 2.
- Raw flattened table: reshape explicitly with `--shape T N C`.
- Raw timestamp/sensor table: pivot externally into `[T, N]` or `[T, N, C]`.
- Raw edge list: convert externally into adjacency matrix `[N, N]`.

### Converting Raw Arrays to `.npz`

Convert a CSV matrix `[T, N]` into canonical `[T, N, 1]`:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.csv \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --add-channel
```

Convert a `.npy` array that is already `[T, N, C]`:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_ARRAY>.npy \
  --output data/<DATASET_NAME>/data.npz \
  --key data
```

Convert a flattened CSV and reshape it:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.csv \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --shape T N C
```

Replace `T`, `N`, and `C` with concrete integers.

Inspect keys in an HDF5 file:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.h5 \
  --list-keys
```

Convert an HDF5 time-series matrix `[T, N]` into `[T, N, 1]`:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.h5 \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --input-key <HDF5_KEY> \
  --add-channel
```

If the HDF5 time-series is already `[T, N, C]`, omit `--add-channel`.

Convert a numeric adjacency matrix:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/adj.csv \
  --output data/<DATASET_NAME>/adj.npz \
  --key adj
```

Convert a pickle adjacency matrix:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/adj.pkl \
  --output data/<DATASET_NAME>/adj.npz \
  --key adj
```

For DCRNN/BasicTS-style adjacency pickles shaped like:

```text
(sensor_ids, sensor_id_to_ind, adj_mx)
```

`--key adj` automatically selects `adj_mx`. For other tuple/list pickle files, pass:

```bash
--pkl-index <INDEX>
```

Inspect the prepared files:

```bash
python scripts/data/inspect_npz.py data/<DATASET_NAME>/data.npz
python scripts/data/inspect_npz.py data/<DATASET_NAME>/adj.npz
```

Expected output should show keys and shapes:

```text
data: shape=(T, N, C), dtype=float32, min=..., max=...
adj: shape=(N, N), dtype=float32, min=..., max=...
```

### Generic BasicTS/DCRNN-Style Workflow

Many traffic datasets are distributed as a time-series file plus an adjacency file:

```text
data/
  raw_data/
    <DATASET_NAME>/
      <TIME_SERIES_FILE>.h5
      <ADJACENCY_FILE>.pkl
      optional_metadata.csv
```

First inspect the HDF5 keys:

```bash
python scripts/data/prepare_npz.py \
  --input data/raw_data/<DATASET_NAME>/<TIME_SERIES_FILE>.h5 \
  --list-keys
```

Then convert the time series. If it is a `[T, N]` matrix, add the channel dimension:

```bash
mkdir -p data/<DATASET_NAME>
python scripts/data/prepare_npz.py \
  --input data/raw_data/<DATASET_NAME>/<TIME_SERIES_FILE>.h5 \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --input-key <HDF5_KEY> \
  --add-channel
```

Convert the adjacency pickle:

```bash
python scripts/data/prepare_npz.py \
  --input data/raw_data/<DATASET_NAME>/<ADJACENCY_FILE>.pkl \
  --output data/<DATASET_NAME>/adj.npz \
  --key adj
```

Check the result:

```bash
python scripts/data/inspect_npz.py data/<DATASET_NAME>/data.npz
python scripts/data/inspect_npz.py data/<DATASET_NAME>/adj.npz
```

Expected output:

```text
data: shape=(T, N, C), dtype=float32, min=..., max=...
adj: shape=(N, N), dtype=float32, min=..., max=...
```

Run a dry run with the generic file-backed config:

```bash
basicstfm dry-run configs/examples/file_forecasting.yaml \
  --cfg-options \
  data.data_path=data/<DATASET_NAME>/data.npz \
  data.graph_path=data/<DATASET_NAME>/adj.npz \
  model.num_nodes=<N> \
  model.input_dim=<C> \
  model.output_dim=<C>
```

Run a fast CPU smoke test:

```bash
basicstfm train configs/examples/file_forecasting.yaml \
  --cfg-options \
  trainer.device=cpu \
  data.data_path=data/<DATASET_NAME>/data.npz \
  data.graph_path=data/<DATASET_NAME>/adj.npz \
  data.batch_size=8 \
  model.num_nodes=<N> \
  model.input_dim=<C> \
  model.output_dim=<C> \
  pipeline.stages.0.epochs=1
```

Run the normal forecasting example:

```bash
basicstfm train configs/examples/file_forecasting.yaml \
  --cfg-options \
  data.data_path=data/<DATASET_NAME>/data.npz \
  data.graph_path=data/<DATASET_NAME>/adj.npz \
  model.num_nodes=<N> \
  model.input_dim=<C> \
  model.output_dim=<C>
```

Replace `<DATASET_NAME>`, `<HDF5_KEY>`, `<N>`, and `<C>` with the values reported by `inspect_npz.py`.

## Data Interface

Built-in tasks expect batches with the following keys:

- `x`: input tensor with shape `[B, T_in, N, C]`.
- `y`: target tensor with shape `[B, T_out, N, C]`.
- `graph`: optional static graph tensor.

For raw array files, `WindowDataModule` expects data shaped as:

```text
[T, N, C]
```

Where:

- `T` is the number of timesteps.
- `N` is the number of spatial nodes.
- `C` is the number of channels.

Two-dimensional arrays shaped `[T, N]` are automatically expanded to `[T, N, 1]`.

### Using a `.npy` or `.npz` Dataset

Example dataset:

```text
data/my_dataset/data.npz
```

Assume the array is stored under key `data` and has shape `[T, N, C]`.

Config:

```yaml
data:
  type: WindowDataModule
  data_path: data/my_dataset/data.npz
  input_key: data
  input_len: 24
  target_len: 12
  batch_size: 32
  split: [0.7, 0.1, 0.2]
  scaler:
    type: standard
```

If the dataset includes a graph:

```text
data/my_dataset/adj.npz
```

And the adjacency matrix is stored under key `adj`:

```yaml
data:
  type: WindowDataModule
  data_path: data/my_dataset/data.npz
  input_key: data
  graph_path: data/my_dataset/adj.npz
  graph_key: adj
  input_len: 24
  target_len: 12
  batch_size: 32
  split: [0.7, 0.1, 0.2]
```

The model configuration must match the data shape:

```yaml
model:
  type: TinySTFoundationModel
  num_nodes: 207
  input_dim: 1
  output_dim: 1
  input_len: 24
  output_len: 12
```

## Scaling and Rescaling

BasicSTFM now follows the same high-level scale/rescale convention that makes BasicTS convenient for forecasting experiments:

1. The scaler is fitted on the training split only.
2. The dataloader returns raw-value windows.
3. Built-in tasks scale `x` before sending it to the model.
4. Model outputs are inverse-transformed back to the original data scale.
5. Losses and metrics are computed on the original scale.

This design gives the model numerically stable inputs while keeping logged losses and metrics interpretable.

Example config:

```yaml
data:
  type: WindowDataModule
  data_path: data/<DATASET_NAME>/data.npz
  input_key: data
  input_len: 24
  target_len: 12
  scaler:
    type: standard
```

The default `standard` scaler computes channel-wise statistics over the training split:

```text
mean: [1, 1, C]
std:  [1, 1, C]
```

For batch tensors shaped `[B, T, N, C]`, these statistics are broadcast automatically.

To disable scaling:

```yaml
data:
  scaler:
    type: identity
```

## Model Configuration

Built-in models:

- `TinySTFoundationModel`: a compact Transformer-style spatio-temporal backbone.
- `MLPForecaster`: a lightweight node-wise MLP forecaster.

Example `TinySTFoundationModel` config:

```yaml
model:
  type: TinySTFoundationModel
  num_nodes: 32
  input_dim: 2
  output_dim: 2
  input_len: 24
  output_len: 12
  hidden_dim: 64
  num_layers: 2
  num_heads: 4
  ffn_dim: 128
  dropout: 0.1
```

Example `MLPForecaster` config:

```yaml
model:
  type: MLPForecaster
  num_nodes: 32
  input_dim: 2
  output_dim: 2
  input_len: 24
  output_len: 12
  hidden_dim: 128
  num_layers: 2
```

The training command does not change when the model changes:

```bash
basicstfm train configs/examples/forecasting.yaml
```

## Multi-Stage Training

The key abstraction is `pipeline.stages`. Each stage can define its own task, objective, optimizer, scheduler, checkpoint source, and parameter-freezing policy.

General stage template:

```yaml
pipeline:
  stages:
    - name: stage_name
      epochs: 3
      load_from: previous
      strict_load: true
      task:
        type: ForecastingTask
      losses:
        - type: mae
          weight: 1.0
      metrics:
        - type: mae
        - type: rmse
      optimizer:
        type: AdamW
        lr: 0.001
        weight_decay: 0.0001
      scheduler:
        type: CosineAnnealingLR
        T_max: 3
      validate_every: 1
      save_best_by: val/loss/total
      gradient_clip_val: 5.0
```

Field meanings:

- `name`: stage identifier and checkpoint prefix.
- `epochs`: number of epochs for the stage.
- `load_from`: checkpoint path or `previous`.
- `strict_load`: whether checkpoint loading must match exactly.
- `task`: batch-level training flow.
- `losses`: weighted objective list.
- `metrics`: validation and test metrics.
- `optimizer`: PyTorch optimizer configuration.
- `scheduler`: PyTorch learning-rate scheduler configuration.
- `validate_every`: validation frequency in epochs.
- `save_best_by`: validation key used for best-checkpoint selection.
- `gradient_clip_val`: optional gradient clipping threshold.

### Parameter Freezing

Stages support shell-style parameter patterns.

Freeze the encoder and train only selected heads:

```yaml
freeze: ["encoder.*"]
unfreeze: ["forecast_head.*", "norm.*"]
```

Freeze all parameters and unfreeze only the forecasting head:

```yaml
freeze: ["all"]
unfreeze: ["forecast_head.*"]
```

## Custom Components

Custom components are imported through `custom_imports`. Any module listed there is imported before the trainer builds the experiment, allowing registry decorators to run.

```yaml
custom_imports:
  - examples.custom_model
  - examples.custom_loss
```

### Custom Model

Example file:

```text
examples/custom_model.py
```

Minimal pattern:

```python
from torch import nn
from basicstfm.registry import MODELS

@MODELS.register()
class MySTBackbone(nn.Module):
    def forward(self, x, graph=None, mode="forecast"):
        return {"forecast": forecast}
```

Input convention:

```text
x shape = [B, T, N, C]
```

Recommended output convention:

```python
return {
    "forecast": forecast,
    "reconstruction": reconstruction,
    "embedding": embedding,
}
```

Config:

```yaml
custom_imports:
  - examples.custom_model

model:
  type: ResidualMLPForecaster
```

Run the custom-component example:

```bash
basicstfm train configs/examples/custom_components.yaml
```

### Custom Loss

Example file:

```text
examples/custom_loss.py
```

Minimal pattern:

```python
from torch import nn
from basicstfm.registry import LOSSES

@LOSSES.register("my_loss")
class MyLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        return loss
```

Config:

```yaml
custom_imports:
  - examples.custom_loss

pipeline:
  stages:
    - name: custom_forecast
      losses:
        - type: my_loss
          weight: 1.0
```

Built-in losses:

- `mae`
- `mse`
- `huber`

Losses can be combined:

```yaml
losses:
  - type: mae
    weight: 1.0
  - type: mse
    weight: 0.05
```

### Custom Task

A task defines the per-batch behavior of a stage. This is the main mechanism for introducing new training protocols.

Built-in tasks:

- `ForecastingTask`
- `MaskedReconstructionTask`

Example file:

```text
examples/custom_task.py
```

Minimal pattern:

```python
from basicstfm.registry import TASKS
from basicstfm.tasks.base import Task

@TASKS.register()
class MyTask(Task):
    def step(self, model, batch, losses, device):
        return {
            "loss": loss,
            "logs": logs,
            "pred": pred,
            "target": target,
        }
```

Config:

```yaml
custom_imports:
  - examples.custom_task

pipeline:
  stages:
    - name: my_stage
      task:
        type: MyTask
```

The trainer remains generic; task logic determines whether the stage performs forecasting, reconstruction, contrastive learning, distillation, or another custom objective.

## Output Layout

The output directory is controlled by `trainer.work_dir`.

Example:

```yaml
trainer:
  work_dir: runs/synthetic_pretrain_finetune
```

Expected output:

```text
runs/synthetic_pretrain_finetune/
  checkpoints/
    masked_pretrain_best.pt
    masked_pretrain_last.pt
    forecast_finetune_best.pt
    forecast_finetune_last.pt
```

If `work_dir` is omitted, BasicSTFM uses:

```text
runs/<experiment_name>/<timestamp>/
```

## Recommended Conda Workflow

For a fresh clone, run the following commands in order:

```bash
conda create -n basicstfm python=3.10 -y
conda activate basicstfm
git clone https://github.com/<your-org-or-username>/BasicSTFM.git
cd BasicSTFM
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
basicstfm --help
PYTHONPATH=src python -m unittest discover -s tests
basicstfm dry-run configs/examples/multistage_pretrain_finetune.yaml
basicstfm train configs/examples/forecasting.yaml --cfg-options pipeline.stages.0.epochs=1 trainer.device=cpu
basicstfm train configs/examples/multistage_pretrain_finetune.yaml --cfg-options pipeline.stages.0.epochs=1 pipeline.stages.1.epochs=1 trainer.device=cpu
```

If all commands succeed, the environment, CLI, config parser, registry system, trainer, and multi-stage execution path are functional.

## Updating an Existing Clone

To update the repository:

```bash
cd BasicSTFM
git pull --ff-only
conda activate basicstfm
pip install -r requirements.txt
pip install -e .
```

Then rerun the checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests
basicstfm dry-run configs/examples/multistage_pretrain_finetune.yaml
```

## Troubleshooting

### `conda: command not found`

Conda is not available in the current shell. Install Miniconda or Anaconda first, then restart the shell.

If Conda is installed but not initialized, initialize it for Bash:

```bash
conda init bash
source ~/.bashrc
```

Then create the environment again:

```bash
conda create -n basicstfm python=3.10 -y
conda activate basicstfm
```

### `basicstfm` command not found

Activate the Conda environment:

```bash
conda activate basicstfm
```

Reinstall the package:

```bash
pip install -e .
```

Or run with the module entry point:

```bash
PYTHONPATH=src python -m basicstfm.cli --help
```

### `ModuleNotFoundError: No module named 'yaml'`

Install dependencies:

```bash
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'torch'`

Install dependencies:

```bash
pip install -r requirements.txt
```

Then reinstall the package:

```bash
pip install -e .
```

### `RuntimeError: No trainable parameters`

This usually means the current stage freezes all parameters and does not unfreeze any module.

Check:

```yaml
freeze:
unfreeze:
```

### `Expected X nodes, got Y`

The model configuration does not match the dataset node count.

Check:

```yaml
data:
  num_nodes:

model:
  num_nodes:
```

For real datasets, verify that the array shape is `[T, N, C]`.

## Extension Roadmap

Common next steps for research use:

- Add dataset-specific datamodules for benchmark suites.
- Add graph-aware spatio-temporal backbones.
- Add large-scale masked modeling tasks.
- Add contrastive or generative pretraining tasks.
- Add distributed training support.
- Add experiment logging integrations.
- Add benchmark scripts for reproducible comparison tables.

The intended workflow is to express standard experiments through configuration first. When configuration is insufficient, add a registered component and reference it from the YAML file.
