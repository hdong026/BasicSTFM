"""Sentinel / non-finite cleanup + temporal interpolation for multivariate series.

Designed for benchmarks where missing values are encoded as ±9999 (e.g. Weather).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_SENTINELS: tuple[float, ...] = (-9999.0, 9999.0)


def mask_sentinels_and_nonfinite(
    values: np.ndarray,
    *,
    sentinel_values: Sequence[float] = DEFAULT_SENTINELS,
) -> np.ndarray:
    """Return float64 copy with sentinels and non-finite entries set to NaN."""
    x = np.asarray(values, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    for s in sentinel_values:
        bad |= np.isclose(x, float(s)) | (x == float(s))
    x[bad] = np.nan
    return x


def interpolate_columns_linear_then_edge(
    values: np.ndarray,
    *,
    missing_warn_frac: float = 0.3,
    variable_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Per column (variable), interpolate along time (axis 0); fill edge NaNs via nearest valid.

    Parameters
    ----------
    values :
        Shape ``[T, N]`` — timesteps × variables.
    missing_warn_frac :
        If fraction of NaNs **before** interpolation exceeds this for a column, record a warning.

    Returns
    -------
    cleaned :
        ``float32`` array same shape as ``values``.
    meta :
        ``missing_frac_before``, ``warnings``, ``still_nonfinite_columns``.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [T, N] array, got shape {x.shape}")

    t, n = x.shape
    names = (
        [str(variable_names[i]) for i in range(n)]
        if variable_names is not None and len(variable_names) == n
        else [f"var_{i}" for i in range(n)]
    )

    out = np.empty_like(x, dtype=np.float64)
    warnings: List[str] = []
    miss_before: List[float] = []
    still_bad: List[int] = []

    for j in range(n):
        col = x[:, j]
        mfrac = float(np.mean(~np.isfinite(col)))
        miss_before.append(mfrac)
        if mfrac > missing_warn_frac:
            warnings.append(
                f"{names[j]} (index={j}): missing/sentinel fraction {mfrac:.4f} "
                f"before interpolation (threshold={missing_warn_frac})"
            )

        s = pd.Series(col)
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.bfill().ffill()
        arr = s.to_numpy(dtype=np.float64)
        out[:, j] = arr
        if not np.all(np.isfinite(arr)):
            still_bad.append(j)
            warnings.append(
                f"{names[j]} (index={j}): still contains non-finite values after interpolation"
            )

    meta: Dict[str, Any] = {
        "missing_frac_before_per_column": miss_before,
        "warnings": warnings,
        "still_nonfinite_column_indices": still_bad,
    }
    return out.astype(np.float32, copy=False), meta
