# OpenCity Variable-Interface Wrapper

## Overview

This extension keeps the original `OpenCityFoundationModel` unchanged and wraps it with two lightweight, dataset-aware interface heads:

`arbitrary history -> input interface -> fixed OpenCity backbone -> output interface -> arbitrary horizon`

The backbone remains the shared spatio-temporal core. The new heads absorb dataset-specific interface mismatches such as history length, forecast horizon, and dataset-conditioned feature calibration.

## Variants

- **Variant A: Prototype mixture**
  - learns prototype-conditioned input/output adapters
  - mixes source prototypes for an unseen target from unlabeled target statistics
- **Variant B: Hyper head**
  - maps dataset statistics to low-rank adapter parameters with a small hypernetwork
- **Variant C: Universal modulated**
  - uses one shared head body plus lightweight dataset-conditioned modulation

All three variants use the same wrapper model class:

- `OpenCityVariableInterfaceWrapper`

and the same wrapper-aware task:

- `InterfaceForecastingTask`

Enable them through:

```yaml
custom_imports:
  - basicstfm_ext
```

## How zero-shot works

Zero-shot evaluation does **not** assume a target-specific head checkpoint.

For a held-out target dataset, the wrapper:

1. reads unlabeled context windows from the target datamodule
2. builds a deterministic dataset descriptor from cheap statistics
3. generates or mixes dataset-conditioned interface parameters
4. keeps the shared OpenCity backbone weights frozen unless the stage says otherwise
5. runs forecasting on the target dataset directly

The descriptor uses only cheap statistics such as:

- mean / std / min / max
- lag autocorrelation
- spectral ratios
- missing-value ratio when masks are available
- graph density / average degree / spectral-radius approximation when a graph exists

Descriptor caches can be persisted with `conditioning_cfg.descriptor_cache_path`.

## How few-shot works

Few-shot stages reuse the pretrained wrapper checkpoint from the multisource pretraining stage.

The recommended default is:

- freeze `backbone.*`
- tune only:
  - `input_head.*`
  - `output_head.*`
  - `conditioning.*`

This is expressed through stage `freeze` / `unfreeze` patterns, so it stays fully compatible with the BasicSTFM stage pipeline.

## Backbone loading

The wrapper supports two loading modes:

- `load_method: checkpoint`
  - load a full wrapper checkpoint
- `load_method: load_backbone_weights`
  - load an original OpenCity checkpoint into `wrapper.backbone`

The second mode is useful when you want to warm-start the wrapper from an existing OpenCity backbone checkpoint without any interface-head weights.

## Configs

Three ready-to-run recipes are provided:

- `configs/foundation/opencity_interface_A_prototype_mixture.yaml`
- `configs/foundation/opencity_interface_B_hyper_head.yaml`
- `configs/foundation/opencity_interface_C_universal_modulated.yaml`

These recipes use:

- multisource traffic pretraining on `PEMS-BAY`, `PEMS04`, `PEMS07`
- held-out zero-shot and 5%-few-shot transfer on `METR-LA` and `PEMS08`
- runtime `input_len=24`, `output_len=24`
- fixed OpenCity backbone `input_len=12`, `output_len=12`

## Run commands

Dry-run:

```bash
basicstfm dry-run configs/foundation/opencity_interface_A_prototype_mixture.yaml
basicstfm dry-run configs/foundation/opencity_interface_B_hyper_head.yaml
basicstfm dry-run configs/foundation/opencity_interface_C_universal_modulated.yaml
```

Train:

```bash
basicstfm train configs/foundation/opencity_interface_A_prototype_mixture.yaml
basicstfm train configs/foundation/opencity_interface_B_hyper_head.yaml
basicstfm train configs/foundation/opencity_interface_C_universal_modulated.yaml
```

2-GPU DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 -m basicstfm train \
  configs/foundation/opencity_interface_A_prototype_mixture.yaml \
  --cfg-options trainer.strategy=ddp
```

Swap the config path to run variants B or C.

## Useful knobs

The wrapper exposes the main ablation knobs directly in config:

- `interface_cfg.variant`
- `interface_cfg.head_type: gru | mamba | interp_mlp`
- `interface_cfg.hidden_dim`
- `interface_cfg.num_layers`
- `interface_cfg.bottleneck_dim`
- `interface_cfg.enable_private_branch`
- `interface_cfg.stronger_input_conditioning_than_output`
- `conditioning_cfg.rank`
- `conditioning_cfg.use_graph_stats`
- `conditioning_cfg.use_spectral_stats`
- `conditioning_cfg.zero_shot_init_method`
- `conditioning_cfg.instance_refinement`
- `regularization_cfg.lambda_adv`
- `regularization_cfg.lambda_ortho`
- `regularization_cfg.lambda_red`

If `mamba_ssm` is unavailable, `head_type: mamba` falls back to the GRU backend automatically.
