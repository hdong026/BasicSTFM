#!/usr/bin/env python3
"""Prepare Monash subsets from Zenodo zips, Hugging Face static files (no dataset script), or local archives.

Mapping (URL + ``file_name`` + multivariate layout) matches HuggingFace
``Monash-University/monash_tsf`` ``monash_tsf.py`` builder configs.

**Download acceleration / mirrors**

- Zenodo has no general public mirror; if it is slow, try ``--source hf`` (same zips on the Hub CDN).
- For Hugging Face, set ``MONASH_HF_DATA_BASE`` to a mirror prefix, e.g. for hf-mirror.com::

    export MONASH_HF_DATA_BASE="https://hf-mirror.com/datasets/Monash-University/monash_tsf/resolve/main/data/"
    python scripts/data/prepare_monash15.py --source hf ...

  (Trailing slash optional; must end with ``.../data/`` mirroring the upstream layout.)
- Optional: ``MONASH_ZENODO_ORIGIN`` (default ``https://zenodo.org``) if you route Zenodo via a proxy that keeps the same ``/records/<id>/files/...`` path shape.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.parse
import warnings
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# --- Cleaning (post-parse) ---
ABS_VALUE_CLIP = 1e12  # values beyond become NaN (then impute or drop)
ABS_POST_CLEAN_WARN = 1e9  # warn / smoke-skip if max |x| exceeds this
MIN_FINITE_FRAC = 0.5  # drop series if fewer than this fraction of points are finite before impute


def _parse_tsf_scalar(token: str) -> float:
    v = token.strip()
    if not v or v == "?":
        return float("nan")
    lv = v.lower()
    if lv in ("nan", "na", "null", "none"):
        return float("nan")
    try:
        x = float(v)
    except ValueError:
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    if abs(x) > ABS_VALUE_CLIP:
        return float("nan")
    return float(x)


# --- Window counts (mirrors ``basicstfm.data.monash_datamodule`` / ``datamodule._split_lengths``) ---


def _split_lengths(total: int, split: Sequence[float]) -> Tuple[int, int, int]:
    if len(split) != 3:
        raise ValueError("split must contain train/val/test ratios or lengths")
    if all(isinstance(x, float) and x <= 1.0 for x in split):
        train = int(total * split[0])
        val = int(total * split[1])
        test = total - train - val
        return train, val, test
    train, val, test = (int(x) for x in split)
    if train + val + test > total:
        raise ValueError("Explicit split lengths exceed total timesteps")
    return train, val, test


def _resolve_split_edges(length: int, split: Sequence[float]) -> Tuple[int, int, int]:
    tr, va, te = _split_lengths(int(length), split)
    total = tr + va + te
    if length < total:
        raise ValueError(f"series length {length} < split sum {total}")
    te += length - total
    return int(tr), int(va), int(te)


def enumerate_window_placements(
    offsets: np.ndarray,
    lengths: np.ndarray,
    split: Sequence[float],
    input_len: int,
    target_len: int,
    stride: int,
    part: Literal["train", "val", "test"],
) -> List[Tuple[int, int]]:
    wl = int(input_len) + int(target_len)
    stride_v = max(1, int(stride))
    out: List[Tuple[int, int]] = []
    lengths = np.asarray(lengths, dtype=np.int64)
    part_str = str(part)

    for s_idx in range(len(lengths)):
        ln = int(lengths[s_idx])
        tr_len, va_len, te_len = _resolve_split_edges(ln, split)
        if part_str == "train":
            seg_len, seg_start = tr_len, 0
        elif part_str == "val":
            seg_len, seg_start = va_len, tr_len
        else:
            seg_len, seg_start = te_len, tr_len + va_len
        if seg_len < wl:
            continue
        local_last = seg_len - wl
        for ls in range(0, local_last + 1, stride_v):
            out.append((s_idx, seg_start + ls))
    return out


DEFAULT_MONASH15: tuple[str, ...] = (
    "australian_electricity_demand",
    "london_smart_meters",
    "solar_10_minutes",
    "solar_4_seconds",
    "wind_farms_minutely",
    "wind_4_seconds",
    "weather",
    "kdd_cup_2018",
    "temperature_rain",
    "saugeenday",
    "sunspot",
    "pedestrian_counts",
    "kaggle_web_traffic",
    "bitcoin",
    "us_births",
)

_DEFAULT_HF_DATA_BASE = (
    "https://huggingface.co/datasets/Monash-University/monash_tsf/resolve/main/data/"
)


def hf_data_base() -> str:
    """Base URL for ``--source hf`` zip paths; override with env ``MONASH_HF_DATA_BASE``."""
    base = os.environ.get("MONASH_HF_DATA_BASE", _DEFAULT_HF_DATA_BASE).strip()
    if not base.endswith("/"):
        base += "/"
    return base


def _str_to_bool(s: str) -> bool:
    x = str(s).lower()
    if x in ("yes", "true", "t", "1", "y"):
        return True
    if x in ("no", "false", "f", "0", "n"):
        return False
    raise ValueError(f"Invalid boolean meta value: {s!r}")


# --- TSF parsing (aligned with HF ``utils.convert_tsf_to_dataframe``) ---


def convert_tsf_to_dataframe(
    full_file_path_and_name: Union[str, Path],
    replace_missing_vals_with: str = "NaN",
    value_column_name: str = "series_value",
) -> Tuple[pd.DataFrame, Optional[str], Optional[int], Optional[bool], Optional[bool]]:
    col_names: List[str] = []
    col_types: List[str] = []
    all_data: Dict[str, List[Any]] = {}
    line_count = 0
    frequency: Optional[str] = None
    forecast_horizon: Optional[int] = None
    contain_missing_values: Optional[bool] = None
    contain_equal_length: Optional[bool] = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False
    all_series: List[np.ndarray] = []

    path = Path(full_file_path_and_name)
    for encoding in ("cp1252", "utf-8"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            text = ""
    else:
        raise ValueError(f"Could not decode {path} as cp1252 or utf-8")

    for line in text.splitlines():
        line = line.strip()

        if line:
            if line.startswith("@"):
                if not line.startswith("@data"):
                    line_content = line.split(" ")
                    if line.startswith("@attribute"):
                        if len(line_content) != 3:
                            raise ValueError("Invalid meta-data specification.")

                        col_names.append(line_content[1])
                        col_types.append(line_content[2])
                    else:
                        if len(line_content) != 2:
                            raise ValueError("Invalid meta-data specification.")

                        if line.startswith("@frequency"):
                            frequency = line_content[1]
                        elif line.startswith("@horizon"):
                            forecast_horizon = int(line_content[1])
                        elif line.startswith("@missing"):
                            contain_missing_values = _str_to_bool(line_content[1])
                        elif line.startswith("@equallength"):
                            contain_equal_length = _str_to_bool(line_content[1])

                else:
                    if len(col_names) == 0:
                        raise ValueError("Missing attribute section. Attribute section must come before data.")

                    found_data_tag = True
            elif not line.startswith("#"):
                if len(col_names) == 0:
                    raise ValueError("Missing attribute section. Attribute section must come before data.")
                elif not found_data_tag:
                    raise ValueError("Missing @data tag.")
                else:
                    if not started_reading_data_section:
                        started_reading_data_section = True
                        found_data_section = True
                        for col in col_names:
                            all_data[col] = []

                    full_info = line.split(":")

                    if len(full_info) != (len(col_names) + 1):
                        raise ValueError("Missing attributes/values in series.")

                    series_part = full_info[len(full_info) - 1]
                    series_tokens = [t.strip() for t in series_part.split(",")]

                    if len(series_tokens) == 0:
                        raise ValueError(
                            "A given series should contains a set of comma separated numeric values."
                        )

                    numeric_series: List[float] = []
                    for val in series_tokens:
                        numeric_series.append(_parse_tsf_scalar(val))

                    if all(not np.isfinite(x) for x in numeric_series):
                        raise ValueError("All series values are missing.")

                    all_series.append(np.asarray(numeric_series, dtype=np.float64))

                    for i in range(len(col_names)):
                        att_val: Any = None
                        if col_types[i] == "numeric":
                            att_val = int(full_info[i])
                        elif col_types[i] == "string":
                            att_val = str(full_info[i])
                        elif col_types[i] == "date":
                            att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                        else:
                            raise ValueError("Invalid attribute type.")

                        if att_val is None:
                            raise ValueError("Invalid attribute value.")
                        else:
                            all_data[col_names[i]].append(att_val)

        line_count = line_count + 1

    if line_count == 0:
        raise ValueError("Empty file.")
    if len(col_names) == 0:
        raise ValueError("Missing attribute section.")
    if not found_data_section:
        raise ValueError("Missing series information under data section.")

    all_data[value_column_name] = all_series
    loaded_data = pd.DataFrame(all_data)

    return (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    )


# --- Registry from ``monash_tsf.py`` (Zenodo record page + archive name + layout) ---


@dataclass(frozen=True)
class MonashDatasetSpec:
    zenodo_record_url: str
    file_name: str
    multivariate: bool = False
    item_id_column: Optional[Union[str, Tuple[str, ...]]] = None
    data_column: Optional[str] = None
    target_fields: Optional[Tuple[str, ...]] = None
    # We only expand target series into values.npy (not feat_dynamic_real); kept for layout parity.
    feat_dynamic_real_fields: Optional[Tuple[str, ...]] = None
    # After cleaning, optional storage transform (e.g. signed log1p for huge dynamic range).
    value_transform: Optional[str] = None


# fmt: off
MONASH_SPECS: Dict[str, MonashDatasetSpec] = {
    "weather": MonashDatasetSpec("https://zenodo.org/record/4654822", "weather_dataset.zip", data_column="series_type"),
    "tourism_yearly": MonashDatasetSpec("https://zenodo.org/record/4656103", "tourism_yearly_dataset.zip"),
    "tourism_quarterly": MonashDatasetSpec("https://zenodo.org/record/4656093", "tourism_quarterly_dataset.zip"),
    "tourism_monthly": MonashDatasetSpec("https://zenodo.org/record/4656096", "tourism_monthly_dataset.zip"),
    "cif_2016": MonashDatasetSpec("https://zenodo.org/record/4656042", "cif_2016_dataset.zip"),
    "london_smart_meters": MonashDatasetSpec(
        "https://zenodo.org/record/4656072", "london_smart_meters_dataset_with_missing_values.zip"
    ),
    "australian_electricity_demand": MonashDatasetSpec(
        "https://zenodo.org/record/4659727", "australian_electricity_demand_dataset.zip"
    ),
    "wind_farms_minutely": MonashDatasetSpec(
        "https://zenodo.org/record/4654909", "wind_farms_minutely_dataset_with_missing_values.zip"
    ),
    "bitcoin": MonashDatasetSpec(
        "https://zenodo.org/record/5121965",
        "bitcoin_dataset_with_missing_values.zip",
        value_transform="signed_log1p",
    ),
    "pedestrian_counts": MonashDatasetSpec("https://zenodo.org/record/4656626", "pedestrian_counts_dataset.zip"),
    "vehicle_trips": MonashDatasetSpec(
        "https://zenodo.org/record/5122535", "vehicle_trips_dataset_with_missing_values.zip"
    ),
    "kdd_cup_2018": MonashDatasetSpec(
        "https://zenodo.org/record/4656719", "kdd_cup_2018_dataset_with_missing_values.zip"
    ),
    "nn5_daily": MonashDatasetSpec(
        "https://zenodo.org/record/4656110", "nn5_daily_dataset_with_missing_values.zip"
    ),
    "nn5_weekly": MonashDatasetSpec("https://zenodo.org/record/4656125", "nn5_weekly_dataset.zip"),
    "kaggle_web_traffic": MonashDatasetSpec(
        "https://zenodo.org/record/4656080", "kaggle_web_traffic_dataset_with_missing_values.zip"
    ),
    "kaggle_web_traffic_weekly": MonashDatasetSpec(
        "https://zenodo.org/record/4656664", "kaggle_web_traffic_weekly_dataset.zip"
    ),
    "solar_10_minutes": MonashDatasetSpec("https://zenodo.org/record/4656144", "solar_10_minutes_dataset.zip"),
    "solar_weekly": MonashDatasetSpec("https://zenodo.org/record/4656151", "solar_weekly_dataset.zip"),
    "car_parts": MonashDatasetSpec("https://zenodo.org/record/4656022", "car_parts_dataset_with_missing_values.zip"),
    "fred_md": MonashDatasetSpec("https://zenodo.org/record/4654833", "fred_md_dataset.zip"),
    "traffic_hourly": MonashDatasetSpec("https://zenodo.org/record/4656132", "traffic_hourly_dataset.zip"),
    "traffic_weekly": MonashDatasetSpec("https://zenodo.org/record/4656135", "traffic_weekly_dataset.zip"),
    "hospital": MonashDatasetSpec("https://zenodo.org/record/4656014", "hospital_dataset.zip"),
    "covid_deaths": MonashDatasetSpec("https://zenodo.org/record/4656009", "covid_deaths_dataset.zip"),
    "sunspot": MonashDatasetSpec("https://zenodo.org/record/4654773", "sunspot_dataset_with_missing_values.zip"),
    "saugeenday": MonashDatasetSpec("https://zenodo.org/record/4656058", "saugeenday_dataset.zip"),
    "us_births": MonashDatasetSpec("https://zenodo.org/record/4656049", "us_births_dataset.zip"),
    "solar_4_seconds": MonashDatasetSpec("https://zenodo.org/record/4656027", "solar_4_seconds_dataset.zip"),
    "wind_4_seconds": MonashDatasetSpec("https://zenodo.org/record/4656032", "wind_4_seconds_dataset.zip"),
    "rideshare": MonashDatasetSpec(
        "https://zenodo.org/record/5122114",
        "rideshare_dataset_with_missing_values.zip",
        multivariate=True,
        item_id_column=("source_location", "provider_name", "provider_service"),
        data_column="type",
        target_fields=(
            "price_min", "price_mean", "price_max", "distance_min", "distance_mean", "distance_max",
            "surge_min", "surge_mean", "surge_max", "api_calls",
        ),
        feat_dynamic_real_fields=("temp", "rain", "humidity", "clouds", "wind"),
    ),
    "oikolab_weather": MonashDatasetSpec(
        "https://zenodo.org/record/5184708", "oikolab_weather_dataset.zip", data_column="type"
    ),
    "temperature_rain": MonashDatasetSpec(
        "https://zenodo.org/record/5129073",
        "temperature_rain_dataset_with_missing_values.zip",
        multivariate=True,
        item_id_column="station_id",
        data_column="obs_or_fcst",
        target_fields=tuple(
            [x.strip() for x in """
fcst_0_DailyPoP fcst_0_DailyPoP1 fcst_0_DailyPoP10 fcst_0_DailyPoP15 fcst_0_DailyPoP25 fcst_0_DailyPoP5
fcst_0_DailyPoP50 fcst_0_DailyPrecip fcst_0_DailyPrecip10Pct fcst_0_DailyPrecip25Pct fcst_0_DailyPrecip50Pct
fcst_0_DailyPrecip75Pct fcst_1_DailyPoP fcst_1_DailyPoP1 fcst_1_DailyPoP10 fcst_1_DailyPoP15 fcst_1_DailyPoP25
fcst_1_DailyPoP5 fcst_1_DailyPoP50 fcst_1_DailyPrecip fcst_1_DailyPrecip10Pct fcst_1_DailyPrecip25Pct
fcst_1_DailyPrecip50Pct fcst_1_DailyPrecip75Pct fcst_2_DailyPoP fcst_2_DailyPoP1 fcst_2_DailyPoP10 fcst_2_DailyPoP15
fcst_2_DailyPoP25 fcst_2_DailyPoP5 fcst_2_DailyPoP50 fcst_2_DailyPrecip fcst_2_DailyPrecip10Pct fcst_2_DailyPrecip25Pct
fcst_2_DailyPrecip50Pct fcst_2_DailyPrecip75Pct fcst_3_DailyPoP fcst_3_DailyPoP1 fcst_3_DailyPoP10 fcst_3_DailyPoP15
fcst_3_DailyPoP25 fcst_3_DailyPoP5 fcst_3_DailyPoP50 fcst_3_DailyPrecip fcst_3_DailyPrecip10Pct fcst_3_DailyPrecip25Pct
fcst_3_DailyPrecip50Pct fcst_3_DailyPrecip75Pct fcst_4_DailyPoP fcst_4_DailyPoP1 fcst_4_DailyPoP10 fcst_4_DailyPoP15
fcst_4_DailyPoP25 fcst_4_DailyPoP5 fcst_4_DailyPoP50 fcst_4_DailyPrecip fcst_4_DailyPrecip10Pct fcst_4_DailyPrecip25Pct
fcst_4_DailyPrecip50Pct fcst_4_DailyPrecip75Pct fcst_5_DailyPoP fcst_5_DailyPoP1 fcst_5_DailyPoP10 fcst_5_DailyPoP15
fcst_5_DailyPoP25 fcst_5_DailyPoP5 fcst_5_DailyPoP50 fcst_5_DailyPrecip fcst_5_DailyPrecip10Pct fcst_5_DailyPrecip25Pct
fcst_5_DailyPrecip50Pct fcst_5_DailyPrecip75Pct
""".split()]
        ),
        feat_dynamic_real_fields=("T_MEAN", "PRCP_SUM", "T_MAX", "T_MIN"),
    ),
}
# fmt: on


def _normalize_zurl(url: str) -> str:
    return url.strip()


def zenodo_download_url(record_url: str, file_name: str) -> str:
    u = _normalize_zurl(record_url)
    m = re.search(r"zenodo\.org/(?:record|records)/(\d+)", u)
    if not m:
        raise ValueError(f"Not a Zenodo record URL: {record_url!r}")
    rid = m.group(1)
    q = urllib.parse.quote(file_name, safe="")
    origin = os.environ.get("MONASH_ZENODO_ORIGIN", "https://zenodo.org").rstrip("/")
    return f"{origin}/records/{rid}/files/{q}?download=1"


def hf_static_zip_url(file_name: str) -> str:
    return urllib.parse.urljoin(hf_data_base(), urllib.parse.quote(file_name))


def download_file(
    url: str,
    dest: Path,
    *,
    desc: str,
    tqdm_disable: bool,
    skip_existing: bool,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and skip_existing:
        return

    req = urllib.request.Request(url, headers={"User-Agent": "BasicSTFM-prepare_monash15/1.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 — fixed Monash/Zenodo URLs
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None
        bar = None
        if not tqdm_disable and total_i is not None:
            from tqdm.auto import tqdm  # noqa: PLC0415

            bar = tqdm(total=total_i, unit="B", unit_scale=True, desc=desc)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with tmp.open("wb") as f:
                bs = 1024 * 256
                while True:
                    chunk = resp.read(bs)
                    if not chunk:
                        break
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            tmp.replace(dest)
        finally:
            if bar is not None:
                bar.close()
            if tmp.exists() and not dest.exists():
                tmp.unlink(missing_ok=True)


def _find_tsf(extract_root: Path, file_name: str) -> Path:
    stem = Path(file_name).stem
    cand = extract_root / f"{stem}.tsf"
    if cand.is_file():
        return cand
    all_tsf = sorted(extract_root.rglob("*.tsf"))
    if not all_tsf:
        raise FileNotFoundError(f"No .tsf under {extract_root} (expected {stem}.tsf)")
    for p in all_tsf:
        if p.stem == stem:
            return p
    if len(all_tsf) == 1:
        return all_tsf[0]
    raise FileNotFoundError(f"Multiple .tsf files under {extract_root}; expected {stem}.tsf")


def ensure_zip_extracted(
    zip_path: Path,
    extract_root: Path,
    *,
    file_name: str,
    overwrite: bool,
) -> Path:
    marker = extract_root / ".extracted_ok"
    if marker.is_file() and not overwrite:
        return _find_tsf(extract_root, file_name)

    if extract_root.exists() and overwrite:
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    marker.write_text("ok\n", encoding="utf-8")
    return _find_tsf(extract_root, file_name)


def _as_frame(ts: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(ts, pd.DataFrame):
        return ts
    return ts.to_frame().T


def _flatten_target_to_series(target: Any) -> Iterator[np.ndarray]:
    arr = np.asarray(target, dtype=np.float64)
    if arr.size == 0:
        return
    if arr.ndim <= 1:
        yield np.ascontiguousarray(arr.reshape(-1), dtype=np.float64)
        return
    if arr.ndim == 2:
        for row in arr:
            yield np.ascontiguousarray(np.asarray(row, dtype=np.float64).reshape(-1))
        return
    raise ValueError(f"Unsupported target ndim={arr.ndim}: shape={arr.shape}")


def apply_stored_value_transform(arr: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """Optional compressive transform before saving (must match ``meta.cleaning.value_transform``)."""
    if mode is None or mode == "none":
        return np.asarray(arr, dtype=np.float32)
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if mode == "signed_log1p":
        y = np.sign(x) * np.log1p(np.abs(x))
        return np.ascontiguousarray(y, dtype=np.float32)
    raise ValueError(f"unknown value_transform {mode!r}")


def clean_and_filter_series(
    raw_series: List[np.ndarray],
    *,
    min_length: int,
    min_finite_frac: float = MIN_FINITE_FRAC,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    kept: List[np.ndarray] = []
    tot_nan_before = 0
    tot_nan_after = 0
    tot_extreme = 0
    skipped_short = 0
    skipped_sparse = 0
    skipped_interp_fail = 0
    n_raw = len(raw_series)

    for seq in raw_series:
        seq = np.asarray(seq, dtype=np.float64).reshape(-1)
        n = int(seq.size)
        finite = np.isfinite(seq)
        n_fin = int(finite.sum())
        if n < min_length or n_fin < min_length:
            skipped_short += 1
            continue
        finite_frac = n_fin / max(n, 1)
        if finite_frac < float(min_finite_frac):
            skipped_sparse += 1
            continue

        tot_nan_before += int((~finite).sum())
        seq = seq.copy()
        extreme_mask = np.abs(seq) > ABS_VALUE_CLIP
        tot_extreme += int(extreme_mask.sum())
        seq[extreme_mask] = np.nan

        s = pd.Series(seq)
        s2 = s.interpolate(method="linear", limit_direction="both")
        s2 = s2.bfill().ffill()
        arr = s2.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            skipped_interp_fail += 1
            continue
        tot_nan_after += int(np.isnan(arr).sum())
        if int(arr.size) < min_length:
            skipped_short += 1
            continue
        kept.append(np.ascontiguousarray(arr, dtype=np.float32))

    aux: Dict[str, Any] = {
        "n_series_raw": n_raw,
        "n_series_kept": int(len(kept)),
        "skipped_too_short_or_sparse": int(skipped_short),
        "skipped_low_finite_frac": int(skipped_sparse),
        "skipped_interp_failed": int(skipped_interp_fail),
        "n_nan_before": int(tot_nan_before),
        "n_nan_after": int(tot_nan_after),
        "n_extreme_values_removed": int(tot_extreme),
    }
    return kept, aux


def dataset_value_aggregates(values: np.ndarray, lengths: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return {
            "total_points": int(v.size),
            "finite_ratio": 0.0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
            "abs_max": float("nan"),
        }
    return {
        "total_points": int(v.size),
        "finite_ratio": float(fin.size / max(v.size, 1)),
        "min": float(fin.min()),
        "max": float(fin.max()),
        "mean": float(fin.mean()),
        "std": float(fin.std()),
        "p01": float(np.percentile(fin, 1)),
        "p50": float(np.percentile(fin, 50)),
        "p99": float(np.percentile(fin, 99)),
        "abs_max": float(np.max(np.abs(fin))),
    }


def series_list_from_loaded_dataframe(
    loaded_data: pd.DataFrame,
    spec: MonashDatasetSpec,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    out: List[np.ndarray] = []
    aux: Dict[str, Any] = {"layout": "default"}

    if spec.item_id_column is not None:
        cols = spec.item_id_column if isinstance(spec.item_id_column, tuple) else (spec.item_id_column,)
        df = loaded_data.set_index(list(cols)).sort_index()
        dc = spec.data_column
        assert dc is not None and spec.target_fields is not None
        for _cat, item_id in enumerate(df.index.unique()):
            ts = df.loc[item_id]
            ts_df = _as_frame(ts)
            target_fields = ts_df[ts_df[dc].isin(spec.target_fields)]
            target = np.vstack(target_fields["target"].to_list())
            for seq in _flatten_target_to_series(target):
                out.append(seq)
        aux["layout"] = "multivariate_indexed"
        return out, aux

    if spec.target_fields is not None:
        assert spec.data_column is not None
        dc = spec.data_column
        sub = loaded_data[loaded_data[dc].isin(spec.target_fields)]
        for _, ts in sub.iterrows():
            for seq in _flatten_target_to_series(ts["target"]):
                out.append(seq)
        aux["layout"] = "filtered_target_fields"
        return out, aux

    if spec.data_column is not None:
        dc = spec.data_column
        for _, ts in loaded_data.iterrows():
            for seq in _flatten_target_to_series(ts["target"]):
                out.append(seq)
        aux["layout"] = f"data_column:{dc}"
        return out, aux

    for _, ts in loaded_data.iterrows():
        for seq in _flatten_target_to_series(ts["target"]):
            out.append(seq)
    aux["layout"] = "univariate_rows"
    return out, aux


def load_series_for_dataset(
    tsf_path: Path,
    spec: MonashDatasetSpec,
    *,
    min_length: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    loaded, freq, horizon, missing, equallength = convert_tsf_to_dataframe(
        tsf_path, value_column_name="target"
    )
    raw_series, layout_extra = series_list_from_loaded_dataframe(loaded, spec)
    kept, clean_stats = clean_and_filter_series(raw_series, min_length=min_length)

    meta_footer: Dict[str, Any] = {
        "tsf_frequency": freq,
        "tsf_horizon": horizon,
        "tsf_missing_flag": missing,
        "tsf_equallength": equallength,
        **clean_stats,
        **layout_extra,
    }
    return kept, meta_footer


def materialize_arrays(series_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not series_list:
        values = np.zeros(0, dtype=np.float32)
        lengths = np.zeros(0, dtype=np.int64)
        offsets = np.zeros(0, dtype=np.int64)
        return values, offsets, lengths
    values = np.concatenate(series_list).astype(np.float32, copy=False)
    lengths = np.asarray([len(s) for s in series_list], dtype=np.int64)
    offsets = np.zeros_like(lengths)
    ofs = 0
    for i in range(len(lengths)):
        offsets[i] = ofs
        ofs += int(lengths[i])
    if ofs != len(values):
        raise RuntimeError("Internal concat offset mismatch")
    return values, offsets, lengths


def obtain_archive_path(
    ds_name: str,
    spec: MonashDatasetSpec,
    source: str,
    cache_dir: Path,
    *,
    tqdm_disable: bool,
    skip_download: bool,
) -> Path:
    file_name = spec.file_name
    dl_dir = cache_dir / "downloads"
    zip_path = dl_dir / file_name

    if source == "local":
        cand = [cache_dir / file_name, cache_dir / ds_name / file_name, Path(file_name)]
        for c in cand:
            if c.is_file():
                return c.resolve()
        raise FileNotFoundError(
            f"local source: put {file_name} under {cache_dir!s} or {cache_dir / ds_name!s}, "
            f"or pass absolute path via cache-dir parent layout."
        )

    if source == "zenodo":
        url = zenodo_download_url(spec.zenodo_record_url, file_name)
    elif source == "hf":
        url = hf_static_zip_url(file_name)
    else:
        raise ValueError(f"Unknown source {source!r}")

    download_file(url, zip_path, desc=f"dl:{file_name}", tqdm_disable=tqdm_disable, skip_existing=skip_download)
    return zip_path


def process_one_dataset(
    ds_name: str,
    output_root: Path,
    *,
    source: str,
    cache_dir: Path,
    input_len: int,
    output_len: int,
    stride: int,
    tqdm_disable: bool,
    skip_download: bool,
    overwrite: bool,
    skip_toxic_on_scale: bool,
) -> Optional[Path]:
    spec = MONASH_SPECS.get(ds_name)
    if spec is None:
        raise KeyError(f"Unknown Monash dataset {ds_name!r} (not in MONASH_SPECS)")

    min_len = int(input_len) + int(output_len)
    if min_len <= 0:
        raise ValueError("input_len + output_len must be positive")

    zip_path = obtain_archive_path(
        ds_name,
        spec,
        source,
        cache_dir,
        tqdm_disable=tqdm_disable,
        skip_download=skip_download,
    )
    key = f"{source}_{ds_name}_{spec.file_name}"
    extract_root = cache_dir / "extracted" / key
    tsf_path = ensure_zip_extracted(zip_path, extract_root, file_name=spec.file_name, overwrite=overwrite)

    series_list, tsf_meta = load_series_for_dataset(tsf_path, spec, min_length=min_len)

    vt = spec.value_transform
    if vt:
        series_list = [apply_stored_value_transform(s, vt) for s in series_list]
        tsf_meta = {**tsf_meta, "value_transform": vt}

    dst = output_root / ds_name
    if not series_list:
        msg = f"{ds_name}: no series left after cleaning (see meta for drop counts)"
        if skip_toxic_on_scale:
            warnings.warn(msg, RuntimeWarning)
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            return None
        raise RuntimeError(msg)

    values, offsets, lengths = materialize_arrays(series_list)
    agg = dataset_value_aggregates(values, lengths)
    vw = _estimate_valid_windows(lengths, input_len, output_len, stride)
    abs_mx = float(agg.get("abs_max", float("nan")))

    if abs_mx > ABS_POST_CLEAN_WARN:
        warnings.warn(
            f"{ds_name}: cleaned abs(max)={abs_mx:.4g} exceeds {ABS_POST_CLEAN_WARN:g} — "
            "may harm training; pass --skip-toxic-on-scale to omit this dataset.",
            UserWarning,
        )
        if skip_toxic_on_scale:
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            return None

    dst.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "dataset_name": ds_name,
        "source": source,
        "zenodo_record_url": _normalize_zurl(spec.zenodo_record_url),
        "archive_file": spec.file_name,
        "hf_static_zip": hf_static_zip_url(spec.file_name),
        "n_series_raw": int(tsf_meta.get("n_series_raw", 0)),
        "n_series_kept": int(len(lengths)),
        "n_series": int(len(lengths)),
        "total_timesteps": int(values.size),
        "valid_windows": int(vw),
        "cleaning": {
            "abs_clip_threshold": ABS_VALUE_CLIP,
            "min_finite_frac": MIN_FINITE_FRAC,
            "value_transform": spec.value_transform,
        },
        "distribution": agg,
        **tsf_meta,
    }

    np.save(dst / "values.npy", values)
    np.save(dst / "offsets.npy", offsets)
    np.save(dst / "lengths.npy", lengths)

    with (dst / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    readme = (
        f"# Monash subset: `{ds_name}`\n\n"
        f"- `source`: {source}\n"
        f"- Zenodo: `{_normalize_zurl(spec.zenodo_record_url)}`\n"
        f"- Archive: `{spec.file_name}`\n"
        f"- n_series_raw: {tsf_meta.get('n_series_raw', '?')}\n"
        f"- n_series_kept: {len(lengths)}\n"
        f"- total_points: {agg['total_points']}\n"
        f"- valid_windows (stride={stride}): {vw}\n"
        f"- abs_max: {agg.get('abs_max', 'n/a')}\n"
        f"- mean/std: {agg.get('mean', 'n/a')} / {agg.get('std', 'n/a')}\n"
        f"- p01/p50/p99: {agg.get('p01', 'n/a')} / {agg.get('p50', 'n/a')} / {agg.get('p99', 'n/a')}\n"
        f"- n_nan_before / after: {tsf_meta.get('n_nan_before', '?')} / {tsf_meta.get('n_nan_after', '?')}\n"
        + (
            f"- value_transform: `{vt}`\n" if (vt := tsf_meta.get("value_transform")) else ""
        )
    )
    (dst / "README.md").write_text(readme, encoding="utf-8")

    return dst


def _estimate_valid_windows(
    lengths: np.ndarray,
    input_len: int,
    output_len: int,
    stride: int,
) -> int:
    wl = int(input_len) + int(output_len)
    st = max(1, int(stride))
    n = 0
    for ln in lengths:
        ln = int(ln)
        if ln < wl:
            continue
        n += (ln - wl) // st + 1
    return n


def _estimate_part_windows(
    lengths: np.ndarray,
    *,
    split: Sequence[float],
    input_len: int,
    output_len: int,
    stride: int,
    part: Literal["train", "val", "test"],
) -> int:
    wl = int(input_len) + int(output_len)
    st = max(1, int(stride))
    lengths = np.asarray(lengths, dtype=np.int64)
    part_str = str(part)
    n = 0

    for ln in lengths:
        ln_i = int(ln)
        tr_len, va_len, te_len = _resolve_split_edges(ln_i, split)
        if part_str == "train":
            seg_len = tr_len
        elif part_str == "val":
            seg_len = va_len
        else:
            seg_len = te_len

        if seg_len < wl:
            continue
        n += (seg_len - wl) // st + 1

    return int(n)


def summarize_dir(
    path: Path,
    *,
    input_len: int,
    output_len: int,
    stride: int,
    split: Sequence[float],
) -> Dict[str, Any]:
    values = np.load(path / "values.npy", mmap_mode="r")
    offsets = np.load(path / "offsets.npy")
    lengths = np.load(path / "lengths.npy")

    st = max(1, stride)
    valid_windows = _estimate_valid_windows(lengths, input_len, output_len, st)
    split_t = tuple(split)
    split_windows = {
        "train": _estimate_part_windows(
            lengths,
            split=split_t,
            input_len=input_len,
            output_len=output_len,
            stride=st,
            part="train",
        ),
        "val": _estimate_part_windows(
            lengths,
            split=split_t,
            input_len=input_len,
            output_len=output_len,
            stride=st,
            part="val",
        ),
        "test": _estimate_part_windows(
            lengths,
            split=split_t,
            input_len=input_len,
            output_len=output_len,
            stride=st,
            part="test",
        ),
    }

    payload = {
        "path": str(path),
        "dataset": path.name,
        "n_series": int(len(lengths)),
        "total_points": int(values.size),
        "valid_windows": int(valid_windows),
        "train_windows": int(split_windows["train"]),
        "val_windows": int(split_windows["val"]),
        "test_windows": int(split_windows["test"]),
        "split": list(split),
        "input_len": input_len,
        "output_len": output_len,
        "stride": st,
    }
    meta_path = path / "meta.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            payload["meta_json"] = json.load(f)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_cache = Path.home() / ".cache" / "basicstfm_monash"
    p.add_argument(
        "--source",
        choices=("hf", "zenodo", "local"),
        default="zenodo",
        help=(
            "zenodo: Zenodo files URL; hf: same zips via Hub (set MONASH_HF_DATA_BASE for mirrors); "
            "local: cache-dir"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Monash15"),
        help="Directory to write datasets (each subfolder is one Monash config name)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help="Download/extract cache (and local zip search root when --source local)",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_MONASH15),
        help="Monash builder config names (default: Monash15 list)",
    )
    p.add_argument(
        "--hf-splits",
        nargs="+",
        default=["train"],
        help="[Deprecated] Ignored: full .tsf series are used (HF time splits were script-specific).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="Reuse cached zips without re-downloading")
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only print counts for existing dirs under output-dir (--input/output/stride)",
    )
    p.add_argument("--input-len", type=int, default=96)
    p.add_argument("--output-len", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.1, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Per-series chronological split ratios (matches datamodule Monash loaders)",
    )
    p.add_argument(
        "--skip-toxic-on-scale",
        action="store_true",
        help="Skip writing a dataset when cleaned abs(max) > 1e9 (default: still write; bitcoin uses signed_log1p)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.output_dir)

    skip_toxic_on_scale = bool(args.skip_toxic_on_scale)

    if args.summarize_only:
        summaries = sorted(p for p in root.iterdir() if p.is_dir() and (p / "lengths.npy").exists())
        if not summaries:
            print(json.dumps({"error": "no datasets found", "dir": str(root)}, indent=2))
            return
        for d in summaries:
            info = summarize_dir(
                d,
                input_len=args.input_len,
                output_len=args.output_len,
                stride=args.stride,
                split=tuple(args.split),
            )
            print(json.dumps(info, indent=2, sort_keys=False))
        return

    cache_dir = Path(args.cache_dir)

    if args.hf_splits != ["train"]:
        print(
            "warning: --hf-splits is deprecated and ignored; using full series from each .tsf",
            file=sys.stderr,
        )

    summaries_out: List[Dict[str, Any]] = []

    for name in args.datasets:
        out_dir = root / name
        val_file = out_dir / "values.npy"
        if val_file.exists() and not args.overwrite:
            info = summarize_dir(
                out_dir,
                input_len=args.input_len,
                output_len=args.output_len,
                stride=args.stride,
                split=tuple(args.split),
            )
            summaries_out.append(info)
            continue

        written = process_one_dataset(
            name,
            root,
            source=str(args.source),
            cache_dir=cache_dir,
            input_len=args.input_len,
            output_len=args.output_len,
            stride=args.stride,
            tqdm_disable=args.no_tqdm,
            skip_download=args.skip_download,
            overwrite=args.overwrite,
            skip_toxic_on_scale=skip_toxic_on_scale,
        )
        if written is None:
            print(f"SKIP\t{name}\t(cleaning removed all series or scale guard)", file=sys.stderr)
            continue
        info = summarize_dir(
            written,
            input_len=args.input_len,
            output_len=args.output_len,
            stride=args.stride,
            split=tuple(args.split),
        )
        summaries_out.append(info)

    for row in summaries_out:
        print(
            f"{row['dataset']}\t"
            f"n_series={row['n_series']}\t"
            f"total_points={row['total_points']}\t"
            f"valid_windows={row['valid_windows']}\t"
            f"train={row['train_windows']}\t"
            f"val={row['val_windows']}\t"
            f"test={row['test_windows']}"
        )


if __name__ == "__main__":
    main()
