# OpenCity-D Protocol Adapter

`OpenCity-D` keeps the vendored `OpenCityFoundationModel` untouched and adds a thin protocol layer around it:

`variable input window -> fixed latent slots -> OpenCity backbone -> fixed output slots -> horizon queries -> variable forecast horizon`

## Architecture

The D wrapper is implemented by [opencity_protocol_adapter.py](/Users/donghuanze/Desktop/codex_workspace/BasicSTFM/src/basicstfm_ext/models/opencity_protocol_adapter.py).

It contains four main parts:

1. `input_protocol_adapter`
   - converts arbitrary runtime history length into the backbone's fixed `input_len`
   - default backend: cross-attention latent resampler
   - uses residual identity initialization so `12->12` stays close to vanilla OpenCity

2. `backbone`
   - the original vendored `OpenCityFoundationModel`
   - never modified

3. `output_query_decoder`
   - decodes the backbone's fixed forecast slots into arbitrary horizons
   - default backend: horizon-query cross-attention
   - also uses a small residual scale at initialization

4. `calibration`
   - light dataset conditioning from unlabeled stats only
   - defaults to FiLM/gating, not a heavy dataset-specific head

## Zero-shot

Zero-shot uses the dataset descriptor from unlabeled target windows only.

No target-specific checkpoint is required.

The wrapper computes:

- a dataset descriptor from metadata, raw values, and optional graph stats
- a small calibration embedding
- a protocol-aligned forecast through the fixed OpenCity backbone

## Few-shot

The default D few-shot scope is:

- `input_protocol_adapter.*`
- `output_query_decoder.*`
- `calibration.*`
- `backbone.forecast_head.*`

This is more fair than adapter-only tuning because the protocol layer changes how the backbone forecast head is exercised.

## Length curriculum

The curriculum recipe uses `ProtocolForecastingTask` to sample shorter runtime lengths from a larger fixed training window.

This keeps the existing datamodule intact while allowing:

- `12 -> 12`
- `24 -> 12`
- `12 -> 24`
- mixed pretraining windows such as `12/24`

The current minimal implementation uses task-level length slicing rather than a new datamodule.

## Configs

Matched-length main recipe:

- [opencity_interface_D_protocol_adapter.yaml](/Users/donghuanze/Desktop/codex_workspace/BasicSTFM/configs/foundation/opencity_interface_D_protocol_adapter.yaml)

Matched-length fair few-shot comparison:

- [opencity_interface_D_protocol_adapter_fair_fewshot.yaml](/Users/donghuanze/Desktop/codex_workspace/BasicSTFM/configs/foundation/opencity_interface_D_protocol_adapter_fair_fewshot.yaml)

Length-generalization curriculum:

- [opencity_interface_D_protocol_adapter_length_curriculum.yaml](/Users/donghuanze/Desktop/codex_workspace/BasicSTFM/configs/foundation/opencity_interface_D_protocol_adapter_length_curriculum.yaml)

## How To Run

Dry-run:

```bash
basicstfm dry-run configs/foundation/opencity_interface_D_protocol_adapter.yaml
basicstfm dry-run configs/foundation/opencity_interface_D_protocol_adapter_fair_fewshot.yaml
basicstfm dry-run configs/foundation/opencity_interface_D_protocol_adapter_length_curriculum.yaml
```

Matched-length pretrain + zero/few-shot:

```bash
basicstfm train configs/foundation/opencity_interface_D_protocol_adapter.yaml
```

Fair few-shot matched-length comparison:

```bash
basicstfm train configs/foundation/opencity_interface_D_protocol_adapter_fair_fewshot.yaml
```

Length-generalization curriculum:

```bash
basicstfm train configs/foundation/opencity_interface_D_protocol_adapter_length_curriculum.yaml
```

## Smoke Tests

```bash
PYTHONPATH=src python -m unittest \
  tests/test_opencity_D_shapes.py \
  tests/test_opencity_D_checkpoint_loading.py \
  tests/test_opencity_D_identity_mode.py \
  tests/test_opencity_D_fewshot_freeze.py
```

## Notes

- Teacher distillation defaults to the raw backbone path inside the wrapper, so matched-length training learns not to pay unnecessary adapter tax.
- If you already have a vanilla OpenCity checkpoint, you can load it into the wrapper backbone with `load_backbone_weights`.
- The curriculum recipe currently uses `12/24` runtime choices by default. Extending it to `36/48` is a natural next step once compute budget is confirmed.
