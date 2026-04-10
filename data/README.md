# Data Directory

This directory is the default location for datasets used by BasicSTFM experiments.

Large dataset files are intentionally ignored by Git. Keep only lightweight documentation, examples, and metadata in the repository.

## Recommended Layout

Each dataset should live in its own subdirectory:

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

The canonical model input file is:

```text
data/<DATASET_NAME>/data.npz
```

It should contain an array under key `data`:

```text
data: [T, N, C]
```

Where:

- `T`: number of timesteps.
- `N`: number of spatial nodes.
- `C`: number of input channels.

An optional graph file can be stored as:

```text
data/<DATASET_NAME>/adj.npz
```

It should contain an array under key `adj`:

```text
adj: [N, N]
```

## Downloading Datasets

BasicSTFM does not vendor benchmark datasets. Download datasets from their official provider, benchmark repository, or your lab storage.

For datasets commonly used with BasicTS-style traffic forecasting experiments, consult the data preparation links in the [BasicTS repository](https://github.com/GestaltCogTeam/BasicTS) and place the downloaded files under `data/<DATASET_NAME>/raw/` before conversion.

Recommended workflow:

```bash
mkdir -p data/<DATASET_NAME>/raw
wget -O data/<DATASET_NAME>/raw/<RAW_FILE_NAME> "<DATASET_URL>"
```

If your data is already in `.npy` or `.npz` format, convert it to the canonical layout:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_ARRAY_FILE> \
  --output data/<DATASET_NAME>/data.npz \
  --key data
```

If your data is a CSV matrix shaped `[T, N]`, convert it with:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.csv \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --add-channel
```

If your data is a CSV tensor flattened as columns, reshape it explicitly:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.csv \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --shape T N C
```

Replace `T`, `N`, and `C` with concrete integers.

If your data is an HDF5 file, inspect its keys first:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.h5 \
  --list-keys
```

Then convert it:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<RAW_FILE>.h5 \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --input-key <HDF5_KEY> \
  --add-channel
```

If the HDF5 array is already `[T, N, C]`, omit `--add-channel`.

If your adjacency file is a DCRNN/BasicTS-style pickle tuple such as:

```text
(sensor_ids, sensor_id_to_ind, adj_mx)
```

Convert it with:

```bash
python scripts/data/prepare_npz.py \
  --input data/<DATASET_NAME>/raw/<ADJ_FILE>.pkl \
  --output data/<DATASET_NAME>/adj.npz \
  --key adj
```

For other tuple/list pickle files, pass `--pkl-index <INDEX>`.

## Using a Dataset in Config

```yaml
data:
  type: WindowDataModule
  data_path: data/<DATASET_NAME>/data.npz
  input_key: data
  graph_path: data/<DATASET_NAME>/adj.npz
  graph_key: adj
  input_len: 24
  output_len: 12
  batch_size: 32
  split: [0.7, 0.1, 0.2]
  scaler:
    type: standard
```

If no graph is available, remove `graph_path` and `graph_key`.

## Generic Time-Series Plus Adjacency Example

If your raw directory looks like:

```text
data/raw_data/<DATASET_NAME>/
  <TIME_SERIES_FILE>.h5
  <ADJACENCY_FILE>.pkl
  optional_metadata.csv
```

Prepare the time series:

```bash
mkdir -p data/<DATASET_NAME>
python scripts/data/prepare_npz.py \
  --input data/raw_data/<DATASET_NAME>/<TIME_SERIES_FILE>.h5 \
  --output data/<DATASET_NAME>/data.npz \
  --key data \
  --input-key <HDF5_KEY> \
  --add-channel
```

Prepare the adjacency matrix:

```bash
python scripts/data/prepare_npz.py \
  --input data/raw_data/<DATASET_NAME>/<ADJACENCY_FILE>.pkl \
  --output data/<DATASET_NAME>/adj.npz \
  --key adj
```

Inspect:

```bash
python scripts/data/inspect_npz.py data/<DATASET_NAME>/data.npz
python scripts/data/inspect_npz.py data/<DATASET_NAME>/adj.npz
```

Expected shapes:

```text
data: shape=(T, N, C)
adj: shape=(N, N)
```

Then run:

```bash
basicstfm train configs/examples/file_forecasting.yaml \
  --cfg-options \
  data.data_path=data/<DATASET_NAME>/data.npz \
  data.graph_path=data/<DATASET_NAME>/adj.npz
```

The default model config uses `auto` for `num_nodes`, `input_dim`, `output_dim`, `input_len`, and `output_len`, so these values are inferred from the prepared data and datamodule settings.

To change the forecasting horizon, override only the data window:

```bash
basicstfm train configs/examples/file_forecasting.yaml \
  --cfg-options \
  data.data_path=data/<DATASET_NAME>/data.npz \
  data.graph_path=data/<DATASET_NAME>/adj.npz \
  data.input_len=24 \
  data.output_len=12
```

## Scaling and Rescaling

BasicSTFM follows a BasicTS-style scale/rescale convention:

1. The scaler is fitted on the training split only.
2. Dataloaders return raw-value windows.
3. Built-in tasks scale `x` before it enters the model.
4. Model outputs are inverse-transformed back to the original scale.
5. Losses and metrics are computed on the original scale.

This keeps model optimization numerically stable while preserving interpretable loss and metric values.

## Preprocessing Rule of Thumb

You need preprocessing if your raw data is not already shaped as `[T, N, C]`.

Common cases:

- Raw `[T, N]`: add a channel dimension to get `[T, N, 1]`.
- Raw `[T, N, C]`: save directly as `data.npz` with key `data`.
- Raw table with timestamps and sensor IDs: pivot it externally into `[T, N]` or `[T, N, C]`.
- Raw graph edge list: convert it externally into an adjacency matrix `[N, N]`.

The framework intentionally keeps dataset-specific cleaning outside the core trainer. This avoids hiding benchmark-specific assumptions in generic code.
