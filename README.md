# BasicSTFM

BasicSTFM is a BasicTS-inspired research framework for **spatio-temporal foundation models**. It is built for experiments where the **model**, **task**, **loss**, **training protocol**, and **stage transitions** all need to stay configurable.

The core abstraction is a **stage pipeline**. A recipe is defined as an ordered list of stages, and each stage can choose its own task, model override, data override, checkpoint loading rule, trainable parameters, and evaluation behavior.

## Highlights

- Config-driven experiments with YAML.
- Stage-centric training pipelines for pretraining, zero-shot evaluation, few-shot tuning, and multi-stage transfer.
- Joint multi-dataset pretraining through a configurable multi-loader data interface.
- Built-in OpenCity-, FactoST-, and UniST-style model families.
- Custom models, tasks, and losses through `custom_imports`.
- Generic data preparation scripts for `.h5`, `.npz`, `.npy`, `.pkl`, and `.csv`.
- Checkpointing, resume, artifact passing between stages, and structured result export.
- BasicTS-style scale/rescale workflow for losses and metrics.

## Repository Layout

```text
BasicSTFM/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    examples/
    foundation/
    templates/
  data/
    README.md
  docs/
    stage_recipe_guide.md
  examples/
    custom_model.py
    custom_loss.py
    custom_task.py
  scripts/
    data/
    results/
  src/
    basicstfm/
  tests/
```

Main entry points:

- `configs/foundation/`: built-in STFM recipes.
- `configs/templates/stage_pipeline_template.yaml`: generic starting point for new stage pipelines.
- `configs/examples/custom_stage_recipe.yaml`: end-to-end custom multi-stage example.
- `configs/examples/multi_dataset_pretrain_transfer.yaml`: joint pretraining over multiple datasets, then target-domain transfer.
- `data/README.md`: dataset layout and preprocessing guide.
- `docs/stage_recipe_guide.md`: how to build your own protocol on top of the framework.

## Installation

The recommended environment is **Python 3.10**.

```bash
conda create -n basicstfm python=3.10 -y
conda activate basicstfm

git clone <YOUR_REPOSITORY_URL>
cd BasicSTFM

pip install -r requirements.txt
pip install -e .
```

Verify the CLI:

```bash
which basicstfm
basicstfm --help
```

## Data Preparation

BasicSTFM uses the following canonical dataset layout:

```text
data/
  <DATASET_NAME>/
    data.npz
    adj.npz
```

Expected array shapes:

```text
data: [T, N, C]
adj:  [N, N]   # optional
```

If your raw data is already under `data/raw_data/`, preview the automatic conversion plan:

```bash
python scripts/data/prepare_all.py \
  --raw-root data/raw_data \
  --output-root data \
  --dry-run
```

Convert all supported datasets:

```bash
python scripts/data/prepare_all.py \
  --raw-root data/raw_data \
  --output-root data
```

Inspect the prepared files:

```bash
python scripts/data/inspect_npz.py data/METR-LA/data.npz
python scripts/data/inspect_npz.py data/METR-LA/adj.npz
```

For dataset-specific details, file-key inspection, and manual conversion examples, see [`data/README.md`](data/README.md).

## Quick Start

### 1. Inspect a pipeline

```bash
basicstfm dry-run configs/examples/custom_stage_recipe.yaml
```

### 2. Run a small example

```bash
basicstfm train configs/examples/custom_stage_recipe.yaml
```

### 3. Run a built-in foundation recipe

```bash
basicstfm train configs/foundation/unist_pretrain_zero_fewshot.yaml \
  --cfg-options \
  data.data_path=data/METR-LA/data.npz \
  data.graph_path=data/METR-LA/adj.npz \
  data.input_len=12 \
  data.output_len=12 \
  data.batch_size=16 \
  model.input_len=auto \
  model.output_len=auto
```

### 4. Inspect a multi-dataset pretraining recipe

```bash
basicstfm dry-run configs/examples/multi_dataset_pretrain_transfer.yaml
```

This example uses `MultiDatasetWindowDataModule` for the pretraining stage and
switches back to `WindowDataModule` for downstream target-domain evaluation.

For a ready-to-run UniST version of the same idea:

```bash
basicstfm dry-run configs/foundation/unist_multi_dataset_pretrain_transfer.yaml
```

### 5. Resume

```bash
basicstfm train configs/foundation/unist_pretrain_zero_fewshot.yaml \
  --cfg-options \
  trainer.resume_from=runs/unist_foundation_transfer/checkpoints/latest.pt
```

or:

```bash
basicstfm train configs/foundation/unist_pretrain_zero_fewshot.yaml \
  --cfg-options \
  trainer.auto_resume=true
```

## Built-in Foundation Recipes

| Recipe | Purpose |
| --- | --- |
| `configs/foundation/opencity_pretrain_zero_fewshot.yaml` | OpenCity-style pretraining, zero-shot testing, and efficient head tuning |
| `configs/foundation/factost_pretrain_zero_fewshot.yaml` | FactoST-style UTP followed by factorized spatio-temporal adaptation |
| `configs/foundation/unist_pretrain_zero_fewshot.yaml` | UniST-style masked pretraining followed by prompt-based transfer |
| `configs/foundation/unist_multi_dataset_pretrain_transfer.yaml` | UniST-style joint multi-dataset pretraining, then target-domain zero-shot and prompt tuning |

Current built-in alignment:

- **OpenCity**: pretraining backbone, zero-shot evaluation, efficient head tuning.
- **FactoST**: UTP-style joint temporal pretraining with multi-scale spectral regularization, then STA-style transfer with metadata fusion, relation filtering, and prototype refinement.
- **UniST**: mixed masking during stage-1 pretraining, then prompt-enabled masked future completion for zero-shot and few-shot transfer.

These recipes are designed to be **stage-faithful abstractions**, not literal line-by-line copies of the original training codebases.

The original OpenCity, FactoST, and UniST papers all rely on multi-dataset
joint pretraining. In BasicSTFM, that pattern is expressed through a
stage-level `data` override that uses `MultiDatasetWindowDataModule`.

## Stage Pipeline Abstraction

Each stage can define:

- `task`
- `model`
- `data`
- `load_from`
- `load_method`
- `save_artifact`
- `freeze` / `unfreeze`
- `eval_only`
- `few_shot_ratio` / `few_shot_windows`
- `reset_model` / `reset_data`

For data, BasicSTFM supports both:

- `WindowDataModule` for a single target dataset;
- `MultiDatasetWindowDataModule` for joint pretraining over multiple datasets.

The multi-dataset module exposes:

- `datasets`: list of dataset entries (`name`, `data_path`, `graph_path`, and optional overrides);
- `train_strategy`: `round_robin`, `proportional`, `uniform`, or `sequential`;
- `eval_strategy`: `per_dataset` or `combined`.

Minimal example:

```yaml
pipeline:
  stages:
    - name: pretrain
      save_artifact: backbone
      epochs: 5
      task:
        type: MaskedReconstructionTask

    - name: zero_shot_eval
      eval_only: true
      epochs: 0
      load_from: backbone
      task:
        type: ForecastingTask

    - name: few_shot_tuning
      epochs: 3
      load_from: backbone
      few_shot_ratio: 0.05
      freeze: [all]
      unfreeze: [forecast_head.*]
      task:
        type: ForecastingTask
```

Important rules:

- Use `reset_model: true` when a stage changes `model.type`.
- Use `reset_data: true` when a stage changes `data.type`.
- Use `load_method` when a stage should only load part of a checkpoint, such as a backbone-only transfer.

See:

- [`configs/templates/stage_pipeline_template.yaml`](configs/templates/stage_pipeline_template.yaml)
- [`docs/stage_recipe_guide.md`](docs/stage_recipe_guide.md)

## Customization

Register your own components and load them through `custom_imports`:

```yaml
custom_imports:
  - examples.custom_model
  - examples.custom_task
  - examples.custom_loss
  - my_project.models
  - my_project.tasks
  - my_project.losses
```

Useful reference files:

- [`examples/custom_model.py`](examples/custom_model.py)
- [`examples/custom_task.py`](examples/custom_task.py)
- [`examples/custom_loss.py`](examples/custom_loss.py)
- [`configs/examples/custom_stage_recipe.yaml`](configs/examples/custom_stage_recipe.yaml)

## Results and Checkpoints

Checkpoints are stored under:

```text
runs/<experiment_name>/checkpoints/
```

Structured stage results are stored under:

```text
runs/<experiment_name>/results/stage_results.json
```

This file records:

- stage name and index
- task, model type, and data type
- load strategy and artifact alias
- resolved stage configs
- train/validation/test metrics
- best score and latest checkpoint path

To collect multiple runs into a benchmark table:

```bash
python scripts/results/export_results.py \
  --input-roots runs \
  --split test \
  --metrics metric/mae metric/rmse \
  --flat-output benchmark/all_stage_results.csv \
  --table-output benchmark/test_metrics.md
```

This scans every `stage_results.json` under `runs/`, exports a flat CSV, and
creates a Markdown summary table for quick comparison.

## Notes

- OpenCity reference code is vendored under its MIT license.
- FactoST and UniST are provided as framework-native adapters rather than large vendored copies.
- Full dataset instructions are intentionally kept outside the main README. Use [`data/README.md`](data/README.md) for preprocessing details.
- Full protocol-extension notes are in [`docs/stage_recipe_guide.md`](docs/stage_recipe_guide.md).
