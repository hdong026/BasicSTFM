# Cross-domain pretraining and METR-LA transfer

Recipes in this directory share one **cross-domain** setup: four sources (LargeST,
KnowAir, Weather, ETTm1), **round_robin** training mixing, fixed **steps_per_epoch**,
**per_dataset** validation, and per-dataset **max_train_windows** caps in
`dataset_registry` (see `MultiDatasetWindowDataModule` in `src/basicstfm/data/datamodule.py`).

The paragraph above matches the **older `*_pretrain_transfer.yaml` recipes** in this
folder (e.g. four sources, often documented as 96 → 96 and **METR-LA** as the transfer
target in prose).

**Protocol note (sharded transfer recipes):** Files such as
`*_cross_domain_sharded_transfer.yaml` use **input / output length 12 → 12** on transfer
stages, **split** `[0.7, 0.1, 0.2]`, and **five** downstream targets: **METR-LA**,
**PEMS04**, **PEMS07**, **PEMS08**, and **ETTm2** (each with zero-shot eval and 5%
few-shot). By default, those target `WindowDataModule` blocks set
`max_val_windows` / `max_test_windows` to speed up dev runs (subset **val/test**
windows after the time split). For **full** val/test on the 0.1 / 0.2 time splits, use
the **P0** audit copies `*_cross_domain_sharded_transfer_p0_full_eval.yaml` (and
`dpm_stfm_v4_cross_domain_e2e_transfer_p0_full_eval.yaml` for the e2e v4 line), one per
family: **OpenCity**, **FactoST**, **UniST**, **DPM v2 / v3 / DPM-SR**, plus v4 e2e.
There is also `dpm_v3_cross_domain_sharded_transfer_p0_sanity.yaml` (PEMS04 + METR-LA
only). See comments in each file and `results/p0_eval_protocol_audit.jsonl` from the
trainer for per-stage window counts. Pretrain blocks are unchanged vs the matching
non-P0 recipe; only target stages drop the val/test caps.

---

## 1. Prerequisites

From the repository root (replace the path with your clone):

```bash
conda activate basicstfm
cd /path/to/BasicSTFM
```

Install the package if you have not already (`pip install -e .`). Data are expected under
`data/<DatasetName>/` as in the top-level `data/README.md`.

---

## 2. Config files (short map)

| Path (under `configs/cross_domain/` unless noted) | Model | Pretrain → transfer (summary) |
|---------------------------------------------------|--------|-------------------------------|
| `dpm_v3_cross_domain_pretrain_transfer.yaml` | DPMV3 + `StableResidualForecastingTaskV3` (Stage I) | 3 DPM stages + ZS + 5% FS |
| `dpm_v2_cross_domain_pretrain_transfer.yaml` | DPMV2 | 3 DPM stages + ZS + 5% FS |
| `dpm_stfm_cross_domain_pretrain_transfer.yaml` | SRDSTFM (“DPM-SR”) | 3 DPM stages + ZS + 5% FS |
| `opencity_cross_domain_pretrain_transfer.yaml` | OpenCity | supervised pretrain + ZS + head FS |
| `factost_cross_domain_pretrain_transfer.yaml` | FactoST | UTP + ZS + adapter FS |
| `unist_cross_domain_pretrain_transfer.yaml` | UniST | masked pretrain + ZS + prompt FS |

**Duplicate DPM v3 path (keep in sync if you edit both):**

`configs/foundation/dpm_cross_domain_pretrain_transfer.yaml` — same protocol as
`dpm_v3_cross_domain_pretrain_transfer.yaml`, but `experiment_name` /
`trainer.work_dir` default to `dpm_cross_domain_pretrain_transfer` and
`runs/dpm_cross_domain_pretrain_transfer/`.

---

## 3. Inspect the resolved recipe (optional)

Prints the fully expanded YAML (registry, `dataset_group` → `datasets`, etc.):

```bash
basicstfm print-config configs/cross_domain/dpm_v3_cross_domain_pretrain_transfer.yaml
```

Override any key at launch, e.g.:

```bash
basicstfm train configs/cross_domain/dpm_v3_cross_domain_pretrain_transfer.yaml \
  --cfg-options trainer.work_dir=runs/my_ablation seed=0
```

---

## 4. Dry-run (stage plan only)

No model build, no data load. Validates the pipeline graph and stage wiring.

```bash
basicstfm dry-run configs/cross_domain/dpm_v3_cross_domain_pretrain_transfer.yaml
basicstfm dry-run configs/cross_domain/dpm_v2_cross_domain_pretrain_transfer.yaml
basicstfm dry-run configs/cross_domain/dpm_stfm_cross_domain_pretrain_transfer.yaml
basicstfm dry-run configs/cross_domain/opencity_cross_domain_pretrain_transfer.yaml
basicstfm dry-run configs/cross_domain/factost_cross_domain_pretrain_transfer.yaml
basicstfm dry-run configs/cross_domain/unist_cross_domain_pretrain_transfer.yaml
```

Optional (legacy DPM v3 duplicate):

```bash
basicstfm dry-run configs/foundation/dpm_cross_domain_pretrain_transfer.yaml
```

---

## 5. Full training (pretrain + METR-LA eval + few-shot)

Each run writes under `runs/<experiment_name>/` (see `trainer.work_dir` in the YAML) and
exports `stage_results.json` when stages complete.

**All six recipes in one block (run separately or as a shell loop):**

```bash
basicstfm train configs/cross_domain/dpm_v3_cross_domain_pretrain_transfer.yaml
basicstfm train configs/cross_domain/dpm_v2_cross_domain_pretrain_transfer.yaml
basicstfm train configs/cross_domain/dpm_stfm_cross_domain_pretrain_transfer.yaml
basicstfm train configs/cross_domain/opencity_cross_domain_pretrain_transfer.yaml
basicstfm train configs/cross_domain/factost_cross_domain_pretrain_transfer.yaml
basicstfm train configs/cross_domain/unist_cross_domain_pretrain_transfer.yaml
```

**Resume** (if your trainer config supports it), for example from the last checkpoint:

```bash
basicstfm train configs/cross_domain/dpm_v3_cross_domain_pretrain_transfer.yaml \
  --cfg-options trainer.resume_from=runs/dpm_v3_cross_domain_pretrain_transfer/checkpoints/latest.pt
```

(Adjust the checkpoint path to the file you actually have.)

**Single-node multi-GPU (DDP)** example — two processes:

```bash
torchrun --nproc_per_node=2 -m basicstfm train \
  configs/cross_domain/dpm_v3_cross_domain_pretrain_transfer.yaml \
  --cfg-options trainer.strategy=ddp
```

---

## 6. Visualization and tables (after runs finish)

Aggregate **zero-shot** and **few-shot** rows from `stage_results.json` under each run
directory. Requires `matplotlib` for figures.

**Single command** (all six cross-domain runs; narrow to METR-LA test MAE):

```bash
python scripts/results/visualize_benchmark.py \
  --input-roots \
    runs/dpm_v3_cross_domain_pretrain_transfer \
    runs/dpm_v2_cross_domain_pretrain_transfer \
    runs/dpm_stfm_cross_domain_pretrain_transfer \
    runs/opencity_cross_domain_pretrain_transfer \
    runs/factost_cross_domain_pretrain_transfer \
    runs/unist_cross_domain_pretrain_transfer \
  --datasets METR-LA \
  --metric metric/mae \
  --split test \
  --output-dir benchmark/paper \
  --prefix cross_domain_96x96_metr \
  --title "Cross-domain pretrain → METR-LA (96→96)" \
  --model-order \
    "OpenCity (XD)" "FactoST (XD)" "UniST (XD)" "DPM-SR (XD)" "DPM-v2 (XD)" "DPM-v3 (XD)"
```

- **`--input-roots`**: directories to search for `stage_results.json`, or explicit paths to
  that file.
- **`--metric`**: e.g. `metric/mae` or `metric/rmse` (see exported keys in your results).
- **`--output-dir` / `--prefix`**: where CSV/figure names are written.
- **`--model-order`**: legend/table order; display names for these recipes are defined in
  `pretty_model_name` in `src/basicstfm/utils/results.py` (e.g. `DPM-v3 (XD)`).

If you also used the **foundation** duplicate for v3, add that run root to `--input-roots`
or merge results manually — avoid training both duplicates to the same metric row unless
you rename `experiment_name`.

---

## 7. Adding ETTm2 (five sources)

1. Set `dataset_group: cross_domain_sources_plus_ettm2` in every **pretrain** stage that
   uses `MultiDatasetWindowDataModule`.
2. For **DPM-family** models only: set `model.num_datasets: 5`.
3. Prefer `steps_per_epoch` divisible by **5** so round-robin steps per domain stay even.

---

## 8. Run directory names (default `trainer.work_dir`)

| Config | Default work directory under `runs/` |
|--------|--------------------------------------|
| `dpm_v3_cross_domain_pretrain_transfer.yaml` | `dpm_v3_cross_domain_pretrain_transfer` |
| `dpm_v2_cross_domain_pretrain_transfer.yaml` | `dpm_v2_cross_domain_pretrain_transfer` |
| `dpm_stfm_cross_domain_pretrain_transfer.yaml` | `dpm_stfm_cross_domain_pretrain_transfer` |
| `opencity_cross_domain_pretrain_transfer.yaml` | `opencity_cross_domain_pretrain_transfer` |
| `factost_cross_domain_pretrain_transfer.yaml` | `factost_cross_domain_pretrain_transfer` |
| `unist_cross_domain_pretrain_transfer.yaml` | `unist_cross_domain_pretrain_transfer` |
| `configs/foundation/dpm_cross_domain_pretrain_transfer.yaml` | `dpm_cross_domain_pretrain_transfer` |

Use these paths in `--input-roots` for `visualize_benchmark.py` and in `resume_from`
overrides.
