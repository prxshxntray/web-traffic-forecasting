from dataclasses import dataclass
import numpy as np
import pandas as pd
import gc

def detect_date_columns(df: pd.DataFrame, id_col: str = "Page") -> list[str]:
    cols = [c for c in df.columns if c != id_col]
    parsed = pd.to_datetime(cols, errors="coerce")
    date_cols = [c for c, d in zip(cols, parsed) if not pd.isna(d)]
    return sorted(date_cols, key=lambda c: pd.to_datetime(c))

@dataclass
class BaselineStats:
    pages: np.ndarray
    last_date: pd.Timestamp
    last28_median: np.ndarray
    weekday_median: np.ndarray
    global_median: float

def build_baseline_stats(
    train_df: pd.DataFrame,
    id_col: str = "Page",
    lookback_weekday: int = 56,
    lookback_level: int = 28,
) -> BaselineStats:

    date_cols = detect_date_columns(train_df, id_col=id_col)
    if len(date_cols) < max(lookback_weekday, lookback_level):
        raise ValueError(f"Not enough date columns ({len(date_cols)}) for lookback windows.")

    pages = train_df[id_col].astype(str).values
    dates = pd.to_datetime(date_cols)
    last_date = dates.max()

    y = train_df[date_cols].to_numpy(dtype=np.float32, copy=True)
    y[y < 0] = np.nan
    y_log = np.log1p(y)

    idx_level_start = len(date_cols) - lookback_level
    last28_median = np.nanmedian(y_log[:, idx_level_start:], axis=1)

    idx_wk_start = len(date_cols) - lookback_weekday
    recent_dates = dates[idx_wk_start:]
    recent_y = y_log[:, idx_wk_start:]

    weekday_median = np.full((recent_y.shape[0], 7), np.nan, dtype=np.float32)
    for wd in range(7):
        mask = (recent_dates.weekday == wd)
        if mask.sum() == 0:
            continue
        weekday_median[:, wd] = np.nanmedian(recent_y[:, mask], axis=1)

    global_median = float(np.nanmedian(y_log))

    del y, y_log, recent_y
    gc.collect()

    return BaselineStats(
        pages=pages,
        last_date=last_date,
        last28_median=last28_median.astype(np.float32),
        weekday_median=weekday_median.astype(np.float32),
        global_median=global_median,
    )

def predict_with_baseline(stats: BaselineStats, key_df: pd.DataFrame, alpha_weekday: float = 0.7) -> pd.DataFrame:
    page_to_idx = pd.Series(np.arange(len(stats.pages)), index=stats.pages)

    key = key_df.copy()
    key["idx"] = page_to_idx.reindex(key["page"]).to_numpy()

    wd = key["date"].dt.weekday.to_numpy(dtype=np.int16)

    pred_log = np.full(len(key), stats.global_median, dtype=np.float32)

    valid = ~pd.isna(key["idx"].to_numpy())
    if valid.any():
        idx = key.loc[valid, "idx"].astype(np.int64).to_numpy()
        wd_valid = wd[valid]

        wmed = stats.weekday_median[idx, wd_valid]
        lmed = stats.last28_median[idx]

        wmed_filled = np.where(np.isfinite(wmed), wmed, lmed)
        lmed_filled = np.where(np.isfinite(lmed), lmed, stats.global_median)
        wmed_filled = np.where(np.isfinite(wmed_filled), wmed_filled, lmed_filled)

        pred_log_valid = alpha_weekday * wmed_filled + (1.0 - alpha_weekday) * lmed_filled
        pred_log[valid] = pred_log_valid.astype(np.float32)

    pred = np.expm1(pred_log).astype(np.float32)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    pred = np.clip(pred, 0.0, None)

    return pd.DataFrame({"Id": key["id"], "Visits": pred})
