# Cross-domain dataset scale (BasicSTFM)

Config: `input_len=96`, `output_len=96` → `windows = T - 96 - 96 + 1`.

## Table 1 — measured scale per directory

| Dataset | data file | key | T, N, C | adj file | adj key | adj shape | windows | volume = w×N×C |
|---|---|---|---|---|---|---|---|---|
| LargeST | `data/LargeST/data.npz` | `data` | (525888, 8600, 1) | `data/LargeST/adj.npz` | `adj` | `(8600, 8600)` | 525697 | 4.520994e+09 |
| LargeST_full | `data/LargeST_full/data.npy` | — | (525888, 8600, 1) | `data/LargeST_full/adj.npy` | `adj` | `(8600, 8600)` | 525697 | 4.520994e+09 |
| KnowAir | `data/KnowAir/data.npz` | `data` | (11688, 184, 18) | `data/KnowAir/adj.npz` | `adj` | `(184, 184)` | 11497 | 3.807806e+07 |
| ETTm1 | `data/ETTm1/data.npz` | `data` | (69680, 7, 1) | `data/ETTm1/adj.npz` | `adj` | `(7, 7)` | 69489 | 4.864230e+05 |
| ETTm1_small | `data/ETTm1_small/data.npz` | `data` | (8192, 7, 1) | `data/ETTm1_small/adj.npz` | `adj` | `(7, 7)` | 8001 | 5.600700e+04 |
| ETTm2 | `data/ETTm2/data.npz` | `data` | (69680, 7, 1) | `data/ETTm2/adj.npz` | `adj` | `(7, 7)` | 69489 | 4.864230e+05 |
| ETTm2_small | `data/ETTm2_small/data.npz` | `data` | (8192, 7, 1) | `data/ETTm2_small/adj.npz` | `adj` | `(7, 7)` | 8001 | 5.600700e+04 |
| Weather | `data/Weather/data.npz` | `data` | (52696, 21, 1) | `data/Weather/adj.npz` | `adj` | `(21, 21)` | 52505 | 1.102605e+06 |

## README snippets (if any)

### LargeST

<pre>
# LargeST

Prepared for BasicSTFM by `scripts/data/prepare_all.py`.

## Raw Directory

`data/raw_data/LargeST`

## Selected Time-Series Files

- `ca_his_raw_2017.h5`
- `ca_his_raw_2018.h5`
- `ca_his_raw_2019.h5`
- `ca_his_raw_2020.h5`
- `ca_his_raw_2021.h5`

## Selected Adjacency File

`ca_rn_adj.npy`

## Prepared Files

```text
data.npz
  data: shape=(525888, 8600, 1)

adj.npz
  adj: shape=(8600, 8600)
```

## Run Example

```bash
basicstfm train configs/examples/file_forecasting.yaml \
  --cfg-options \
  data.data_path=data/LargeST/data.npz \
  data.graph_path=data/LargeST/adj.npz \
```
</pre>

### LargeST_full

<pre>
metadata.json:
{
  &quot;source_dir&quot;: &quot;data/raw_data/LargeST&quot;,
  &quot;years&quot;: [
    &quot;2017&quot;,
    &quot;2018&quot;,
    &quot;2019&quot;,
    &quot;2020&quot;,
    &quot;2021&quot;
  ],
  &quot;input_key&quot;: &quot;t&quot;,
  &quot;dtype&quot;: &quot;float32&quot;,
  &quot;data_path&quot;: &quot;data/LargeST_full/data.npy&quot;,
  &quot;adj_path&quot;: &quot;data/LargeST_full/adj.npy&quot;,
  &quot;data_shape&quot;: [
    525888,
    8600,
    1
  ],
  &quot;adj_shape&quot;: [
    8600,
    8600
  ]
}
</pre>

### KnowAir

<pre>
# KnowAir / BasicSTFM 数据说明

## 原始与处理后数组
- 原始 `.npy` shape: `(11688, 184, 18)`
- 处理后 `data.npz[&#x27;data&#x27;]` shape: `(11688, 184, 18)` （期望 `[T, N, C]`）

## 元数据
- 节点粒度判定（auto 启发式）: **city**（N=184 更接近城市 184/196 或站点 1498）
- 实际使用的元数据文件: `/home/dhz/BasicSTFM/data/raw_data/KnowAir/pm25_gnn_metadata/city.txt`
- CLI `--node-level`: `auto`

## 图构造
- kNN k = 8
- 距离: Haversine（km）
- 加权: **高斯核 exp(-d^2/sigma^2)**
- sigma: **124.6462273809926** （km；加权图下 auto 为有向 kNN 边上的距离中位数）
- 对称化: `A = max(A, A.T)`
- 对角线: 置为 1

## 邻接矩阵统计（不含对角线）
- `adj` shape: `(184, 184)`
- 非零率（off-diagonal density）: 0.052744
- 稀疏度（off-diagonal sparsity）: 0.947256
- 平均度（off-diagonal 权重和/节点）: 3.494591
</pre>

### ETTm1

<pre>
# ETTm — BasicSTFM

Prepared by `scripts/data/prepare_ettm_for_basicstfm.py`.

## Source

- CSV: `/home/dhz/BasicSTFM/data/raw_data/ETTm1/ETTm1.csv`
- 时间列: `date`（**不作为节点特征**）
- 节点列（顺序与 `data[..., i, 0]` 一致）: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`

## 数组形状

- 原始表: `69680` 行 × `7` 个变量列（不含时间）
- `data.npz` key `data`: shape `[T, N, C]` = `(69680, 7, 1)`（`C=1`）

## 训练 / 验证 / 测试划分

与 `WindowDataModule` 一致：`train = int(T×0.7)`，`val = int(T×0.1)`，`test` 为剩余。

- train / val / test 长度: **48776 / 6968 / 13936**

## 图构造（**仅用 train 段，无验证/测试泄漏**）

在 train 段上计算变量两两 **Pearson** 相关矩阵 `corr`，按 **|corr|** 为每个节点取 **top-5** 邻居（不含自环），再：

- 对称化：`A = max(A, A^T)`
- 对角线：`1`
- **use_abs_corr=True**：加权版边权为 `|corr|`；二值版为 `0/1`
- **threshold**: `None`（低于阈值的边不参与 top-k 候选）

输出文件：

- `adj_corr_topk.npz` — 加权相关图
- `adj_binary_topk.npz` — 二值 top-k 图
- `adj.npz` — 默认与 **weighted correlation (adj_corr_topk.npz)** 相同

## 统计（非对角）

| 版本 | 非零率 | 平均度 |
|------|--------|--------|
| weighted | 0.952381 | 1.466298 |
| binary | 0.952381 | 5.714286 |

## 复现命令

```bash
python scripts/data/prepare_ettm_for_basicstfm.py \
  --input-csv /home/dhz/BasicSTFM/data/raw_data/ETTm1/ETTm1.csv \
  --output-dir /home/dhz/BasicSTFM/data/ETTm1 \
  --split 0.7 0.1 0.2 \
  --topk 5 \
  --use-abs-corr true \
  --threshold none \
  --binary-graph false
```
</pre>

### ETTm1_small

<pre>
# ETTm — BasicSTFM

Prepared by `scripts/data/prepare_ettm_for_basicstfm.py`.

## Source

- CSV: `/home/dhz/BasicSTFM/data/raw_data/ETTm1/ETTm1.csv`
- 时间列: `date`（**不作为节点特征**）
- 节点列（顺序与 `data[..., i, 0]` 一致）: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`

## 数组形状

- 原始表: `8192` 行 × `7` 个变量列（不含时间）
- `data.npz` key `data`: shape `[T, N, C]` = `(8192, 7, 1)`（`C=1`）

## 训练 / 验证 / 测试划分

与 `WindowDataModule` 一致：`train = int(T×0.7)`，`val = int(T×0.1)`，`test` 为剩余。

- train / val / test 长度: **5734 / 819 / 1639**

## 图构造（**仅用 train 段，无验证/测试泄漏**）

在 train 段上计算变量两两 **Pearson** 相关矩阵 `corr`，按 **|corr|** 为每个节点取 **top-3** 邻居（不含自环），再：

- 对称化：`A = max(A, A^T)`
- 对角线：`1`
- **use_abs_corr=True**：加权版边权为 `|corr|`；二值版为 `0/1`
- **threshold**: `None`（低于阈值的边不参与 top-k 候选）

输出文件：

- `adj_corr_topk.npz` — 加权相关图
- `adj_binary_topk.npz` — 二值 top-k 图
- `adj.npz` — 默认与 **weighted correlation (adj_corr_topk.npz)** 相同

## 统计（非对角）

| 版本 | 非零率 | 平均度 |
|------|--------|--------|
| weighted | 0.619048 | 1.821359 |
| binary | 0.619048 | 3.714286 |

## 复现命令

```bash
python scripts/data/prepare_ettm_for_basicstfm.py \
  --input-csv /home/dhz/BasicSTFM/data/raw_data/ETTm1/ETTm1.csv \
  --output-dir /home/dhz/BasicSTFM/data/ETTm1_small \
  --split 0.7 0.1 0.2 \
  --topk 3 \
  --use-abs-corr true \
  --threshold none \
  --binary-graph false
```
</pre>

### ETTm2

<pre>
# ETTm — BasicSTFM

Prepared by `scripts/data/prepare_ettm_for_basicstfm.py`.

## Source

- CSV: `/home/dhz/BasicSTFM/data/raw_data/ETTm2/ETTm2.csv`
- 时间列: `date`（**不作为节点特征**）
- 节点列（顺序与 `data[..., i, 0]` 一致）: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`

## 数组形状

- 原始表: `69680` 行 × `7` 个变量列（不含时间）
- `data.npz` key `data`: shape `[T, N, C]` = `(69680, 7, 1)`（`C=1`）

## 训练 / 验证 / 测试划分

与 `WindowDataModule` 一致：`train = int(T×0.7)`，`val = int(T×0.1)`，`test` 为剩余。

- train / val / test 长度: **48776 / 6968 / 13936**

## 图构造（**仅用 train 段，无验证/测试泄漏**）

在 train 段上计算变量两两 **Pearson** 相关矩阵 `corr`，按 **|corr|** 为每个节点取 **top-5** 邻居（不含自环），再：

- 对称化：`A = max(A, A^T)`
- 对角线：`1`
- **use_abs_corr=True**：加权版边权为 `|corr|`；二值版为 `0/1`
- **threshold**: `None`（低于阈值的边不参与 top-k 候选）

输出文件：

- `adj_corr_topk.npz` — 加权相关图
- `adj_binary_topk.npz` — 二值 top-k 图
- `adj.npz` — 默认与 **weighted correlation (adj_corr_topk.npz)** 相同

## 统计（非对角）

| 版本 | 非零率 | 平均度 |
|------|--------|--------|
| weighted | 0.857143 | 1.954883 |
| binary | 0.857143 | 5.142857 |

## 复现命令

```bash
python scripts/data/prepare_ettm_for_basicstfm.py \
  --input-csv /home/dhz/BasicSTFM/data/raw_data/ETTm2/ETTm2.csv \
  --output-dir /home/dhz/BasicSTFM/data/ETTm2 \
  --split 0.7 0.1 0.2 \
  --topk 5 \
  --use-abs-corr true \
  --threshold none \
  --binary-graph false
```
</pre>

### ETTm2_small

<pre>
# ETTm — BasicSTFM

Prepared by `scripts/data/prepare_ettm_for_basicstfm.py`.

## Source

- CSV: `/home/dhz/BasicSTFM/data/raw_data/ETTm2/ETTm2.csv`
- 时间列: `date`（**不作为节点特征**）
- 节点列（顺序与 `data[..., i, 0]` 一致）: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`

## 数组形状

- 原始表: `8192` 行 × `7` 个变量列（不含时间）
- `data.npz` key `data`: shape `[T, N, C]` = `(8192, 7, 1)`（`C=1`）

## 训练 / 验证 / 测试划分

与 `WindowDataModule` 一致：`train = int(T×0.7)`，`val = int(T×0.1)`，`test` 为剩余。

- train / val / test 长度: **5734 / 819 / 1639**

## 图构造（**仅用 train 段，无验证/测试泄漏**）

在 train 段上计算变量两两 **Pearson** 相关矩阵 `corr`，按 **|corr|** 为每个节点取 **top-3** 邻居（不含自环），再：

- 对称化：`A = max(A, A^T)`
- 对角线：`1`
- **use_abs_corr=True**：加权版边权为 `|corr|`；二值版为 `0/1`
- **threshold**: `None`（低于阈值的边不参与 top-k 候选）

输出文件：

- `adj_corr_topk.npz` — 加权相关图
- `adj_binary_topk.npz` — 二值 top-k 图
- `adj.npz` — 默认与 **weighted correlation (adj_corr_topk.npz)** 相同

## 统计（非对角）

| 版本 | 非零率 | 平均度 |
|------|--------|--------|
| weighted | 0.666667 | 2.325492 |
| binary | 0.666667 | 4.000000 |

## 复现命令

```bash
python scripts/data/prepare_ettm_for_basicstfm.py \
  --input-csv /home/dhz/BasicSTFM/data/raw_data/ETTm2/ETTm2.csv \
  --output-dir /home/dhz/BasicSTFM/data/ETTm2_small \
  --split 0.7 0.1 0.2 \
  --topk 3 \
  --use-abs-corr true \
  --threshold none \
  --binary-graph false
```
</pre>

### Weather

<pre>
# Weather (TSLib / Autoformer) — BasicSTFM

## 来源

- **Autoformer** 官方 README 数据入口：[Google Drive 六大数据集](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing)
- **Time-Series-Library** 入口：[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) · [Hugging Face 数据集](https://huggingface.co/datasets/thuml/Time-Series-Library)
- 本仓库使用的预处理文件：`weather.csv`（与 TSLib `dataset/weather/` 一致）
- 本地路径：`/home/dhz/BasicSTFM/data/raw_data/Weather/weather.csv`

## 原始 CSV 概要

- **行数（含表头）**：52697；**时间步 T**：52696
- **时间列**：`date`（不作为节点）
- **推断时间粒度**：10-minute
- **变量数 N**：21（每变量一节点，multivariate 设定）
- **列名**：`p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`, `VPmax (mbar)`, `VPact (mbar)`, `VPdef (mbar)`, `sh (g/kg)`, `H2OC (mmol/mol)`, `rho (g/m**3)`, `wv (m/s)`, `max. wv (m/s)`, `wd (deg)`, `rain (mm)`, `raining (s)`, `SWDR (W/m�)`, `PAR (�mol/m�/s)`, `max. PAR (�mol/m�/s)`, `Tlog (degC)`, `OT`

## BasicSTFM 格式

- `data.npz` key `data`：shape **`[52696, 21, 1]`** = `[T, N, C]`

## 划分（与 WindowDataModule 一致）

- `split` = [0.7, 0.1, 0.2] → train / val / test 长度 **36887 / 5269 / 10540**

## 图（可选）

- **已生成** train 段 Pearson top-k 图。
- 若已生成：见 `adj.npz`（默认加权）、`adj_corr_topk.npz`、`adj_binary_topk.npz`、`graph_meta.json`。

## 复现

```bash
python scripts/data/prepare_weather_for_basicstfm.py --download --build-adj --topk 10
```
</pre>

## Table 2 — joint pretrain mixing (heuristic)

In this tree, `LargeST` and `LargeST_full` have the same `(T, N, C)`; `LargeST` is a `.npz` containing a single giant `.npy`, while `LargeST_full` is a standalone `.npy` for mmap. Uncapped, traffic **dominates** `volume` vs air / ETT / weather by orders of magnitude — **cap** traffic windows per epoch when mixing.

### A — prefer `data/LargeST/`

Suggested **max windows / domain / epoch** (tune in your sampler / `steps_per_cap`):

| Domain | cap |
| --- | ---: |
| LargeST | 8192 |
| LargeST_full | 8192 |
| KnowAir | 4096 |
| ETTm1 | 16000 |
| ETTm2 | 16000 |
| Weather | 12000 |
| ETTm1_small | 2000 |
| ETTm2_small | 2000 |

Sampling weights (from **capped** effective `volume`, pow index 0.35): `ETTm1`=0.489 `KnowAir`=0.091 `LargeST`=0.051 `Weather`=0.368.
Effective volumes used: LargeST=7.0451e+07, KnowAir=1.3566e+07, ETTm1=1.1200e+05, Weather=2.5200e+05.

- **ETTm `*_small`**: use for smoke tests / fast debugging; for real pretrain prefer full `ETTm1` or `ETTm2` (more `T`, same `N`).

### B — `data/LargeST_full/`

Same scale as `LargeST` here; I/O may differ. Use **identical** caps. Weights: `ETTm1`=0.489 `KnowAir`=0.091 `LargeST_full`=0.051 `Weather`=0.368.
Effective volumes: LargeST_full=7.0451e+07, KnowAir=1.3566e+07, ETTm1=1.1200e+05, Weather=2.5200e+05.

**Uncapped volume share** (LargeST vs KnowAir+ETTm1+Weather only): **99.1%** in LargeST — mixing **without** caps lets traffic govern gradients.

## Synthesis

1. **First run**: joint pretrain on **`KnowAir` + `Weather` + one ETT (`ETTm1` or `ETTm2`)**; add **`LargeST` (or `LargeST_full`) only with 4k–8k max traffic windows/epoch** or you re-enter a ~**99%**-traffic regime by volume.
2. **`LargeST_full`**: not smaller than `LargeST` in this report — skip only if your pipeline prefers `.npz` zip I/O; otherwise use the **same** caps either way. Do **not** use `full` to “fix” domain imbalance without subsampling.
3. **Slowest** forward/backward: **traffic (N=8600)**. Smallest-`T` ETTm `small` is fast per epoch but **thin** for representation learning.
