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
- Ubuntu-friendly setup with `requirements.txt` and editable installation.

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
  README.md
  requirements.txt
  pyproject.toml
  configs/
    examples/
      forecasting.yaml
      multistage_pretrain_finetune.yaml
      custom_components.yaml
  examples/
    __init__.py
    custom_model.py
    custom_loss.py
    custom_task.py
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
- `examples/custom_model.py`: custom model example.
- `examples/custom_loss.py`: custom loss example.
- `examples/custom_task.py`: custom task example.

## Ubuntu Installation From GitHub

The following instructions assume Ubuntu 22.04 or Ubuntu 24.04. They also work on most recent Debian-based Linux distributions with minor package-name adjustments.

### Step 1: Install System Packages

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential
```

Verify Python:

```bash
python3 --version
```

BasicSTFM requires Python 3.9 or newer.

### Step 2: Clone the Repository

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

### Step 3: Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, your prompt should usually contain `(.venv)`.

Verify the Python interpreter:

```bash
which python
```

Expected output:

```text
.../BasicSTFM/.venv/bin/python
```

### Step 4: Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install Python Dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Then install BasicSTFM in editable mode:

```bash
pip install -e .
```

Editable mode is recommended for research development because changes under `src/basicstfm/` take effect without reinstalling the package.

Alternative one-line developer installation:

```bash
pip install -e ".[dev]"
```

### Step 6: PyTorch Notes

The default `requirements.txt` includes `torch>=2.1`, which is suitable for a basic CPU setup and many standard Linux environments.

If you need a CUDA-specific PyTorch build, install the PyTorch wheel that matches your CUDA driver before installing BasicSTFM dependencies. A common workflow is:

```bash
pip install --upgrade pip setuptools wheel
# Install your CUDA-compatible PyTorch build here.
pip install -r requirements.txt
pip install -e .
```

Check whether PyTorch can see CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

### Step 7: Verify the Command Line Interface

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

## Recommended Ubuntu Workflow

For a fresh Ubuntu machine, run the following commands in order:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential
git clone https://github.com/<your-org-or-username>/BasicSTFM.git
cd BasicSTFM
python3 -m venv .venv
source .venv/bin/activate
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
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then rerun the checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests
basicstfm dry-run configs/examples/multistage_pretrain_finetune.yaml
```

## Troubleshooting

### `python3 -m venv .venv` fails

Install the Ubuntu venv package:

```bash
sudo apt update
sudo apt install -y python3-venv
```

Then recreate the environment:

```bash
python3 -m venv .venv
```

### `basicstfm` command not found

Activate the virtual environment:

```bash
source .venv/bin/activate
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

If you need a CUDA-specific build, install the matching PyTorch wheel for your system, then rerun:

```bash
pip install -r requirements.txt
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
