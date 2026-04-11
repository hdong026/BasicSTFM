# Stage Recipe Guide

BasicSTFM is designed around stage recipes. Instead of writing a new runner for
every model family, you describe your training protocol as an ordered list of
stages.

Each stage can decide:

- which task to run;
- which model or model override to use;
- which data or data override to use;
- which checkpoint or artifact to load;
- how weights should be loaded;
- which parameters are frozen or unfrozen;
- whether the stage trains or only evaluates;
- whether the stage uses a few-shot subset.

This guide is the shortest path from "I have a new spatio-temporal foundation
model idea" to "I can run it inside BasicSTFM."

## 1. Register Your Components

You can keep new research code in your own module and register it through
BasicSTFM's registries.

Minimal examples already exist:

- `examples/custom_model.py`
- `examples/custom_task.py`
- `examples/custom_loss.py`

Load them in the config with:

```yaml
custom_imports:
  - examples.custom_model
  - examples.custom_task
  - examples.custom_loss
  - my_project.models
  - my_project.tasks
  - my_project.losses
```

## 2. Define Base Data and Model Recipes

At the top level of the config, define the default data and model recipe. These
act as the starting point for every stage.

```yaml
data:
  type: WindowDataModule
  data_path: data/METR-LA/data.npz
  graph_path: data/METR-LA/adj.npz
  input_len: 12
  output_len: 12
  batch_size: 32
  split: [0.7, 0.1, 0.2]
  scaler:
    type: standard

model:
  type: TinySTFoundationModel
  num_nodes: auto
  input_dim: auto
  output_dim: auto
  input_len: auto
  output_len: auto
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  ffn_dim: 256
```

If a field is set to `auto`, the trainer resolves it from the active datamodule
metadata for that stage.

## 3. Express the Protocol as Stages

Every experiment must define `pipeline.stages`.

```yaml
pipeline:
  stages:
    - name: backbone_pretrain
      save_artifact: pretrained_backbone
      epochs: 10
      task:
        type: MaskedReconstructionTask
        mask_ratio: 0.5
      losses:
        - type: mse

    - name: zero_shot_eval
      eval_only: true
      epochs: 0
      load_from: pretrained_backbone
      task:
        type: ForecastingTask
      losses:
        - type: mae
      metrics:
        - type: mae
        - type: rmse

    - name: few_shot_tuning
      epochs: 5
      load_from: pretrained_backbone
      few_shot_ratio: 0.05
      freeze: [all]
      unfreeze: [forecast_head.*]
      task:
        type: ForecastingTask
```

That pattern already covers a large fraction of STFM work:

- stage-1 pretraining;
- zero-shot transfer;
- parameter-efficient few-shot tuning.

For prompt-based transfer regimes, you can also use `MaskedForecastCompletionTask`
to mask the future suffix and let the model complete it autoregressively in a
single reconstruction-style forward pass.

## 4. Use Stage-Specific Model or Data Overrides

Some papers switch model behavior between stages without changing the whole
family. In that case, override only the necessary keys.

```yaml
- name: prompt_tuning
  load_from: pretrained_backbone
  model:
    use_prompt: true
  task:
    type: ForecastingTask
    model_mode: prompt_forecast
```

You can also change stage-level dataloader behavior:

```yaml
- name: target_domain_eval
  eval_only: true
  epochs: 0
  data:
    batch_size: 64
```

## 5. Reset Inheritance When Switching Families

If a stage changes `model.type` or `data.type`, use an explicit reset.

```yaml
- name: new_model_family
  reset_model: true
  model:
    type: MyTransferModel
    hidden_dim: 256
```

```yaml
- name: new_dataset_family
  reset_data: true
  data:
    type: MyBenchmarkDataModule
    root: data/target_benchmark
```

BasicSTFM now enforces this explicitly. Without `reset_model: true` or
`reset_data: true`, the trainer raises an error instead of silently inheriting
incompatible keys.

## 6. Choose How Stage Weights Are Loaded

By default, `load_method: checkpoint` restores a normal checkpoint into the
current model.

For STFM protocols that only reuse part of a checkpoint, define a model method
such as `load_backbone_weights` and reference it in the stage:

```yaml
- name: factorized_adapter
  load_from: factost_utp
  load_method: load_backbone_weights
  strict_load: false
  model:
    use_st_adapter: true
```

This is how the built-in FactoST and UniST recipes express stage transitions
that are not simple full-model reloads.

## 7. Freeze Exactly the Parameters You Want

Stage recipes support `freeze` and `unfreeze` patterns using shell-style
matching. `all` is a special shortcut.

```yaml
freeze:
  - all
unfreeze:
  - prompt_*
  - st_*
  - forecast_head.*
```

The trainer reapplies trainability rules at every stage, so each stage starts
from a clean `requires_grad` state before the new freeze plan is applied.

## 8. Keep Evaluation Unified

After every completed stage, BasicSTFM writes structured outputs to:

```text
runs/<experiment_name>/results/stage_results.json
```

The file records, per stage:

- stage name and index;
- task, model type, and data type;
- load strategy and artifact alias;
- resolved model/data configs;
- train/validation/test metrics;
- best score and latest checkpoint path.

This makes it much easier to aggregate results into benchmark tables without
scraping terminal logs.

## 9. Start From the Included Templates

Good starting points:

- `configs/templates/stage_pipeline_template.yaml`
- `configs/examples/custom_stage_recipe.yaml`
- `configs/foundation/factost_pretrain_zero_fewshot.yaml`
- `configs/foundation/unist_pretrain_zero_fewshot.yaml`
- `configs/foundation/opencity_pretrain_zero_fewshot.yaml`

## 10. Recommended Workflow

1. Write or register your model, task, and optional loss.
2. Start from `configs/templates/stage_pipeline_template.yaml`.
3. Replace the synthetic dataset with your benchmark datamodule.
4. Encode your protocol as stages.
5. Run `basicstfm dry-run ...` to inspect the stage plan.
6. Run a 1-epoch smoke test with small batch size.
7. Launch the full experiment.
8. Read `results/stage_results.json` for structured outputs.
9. Export benchmark tables with `python scripts/results/export_results.py ...`.
