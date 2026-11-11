#!/usr/bin/env python3
"""SGCC Dataset Preprocessing & Chronological Alignment Engine.

Standardizes the SGCC electricity theft detection benchmark:
1. Resolves raw unpadded string date columns into chronological calendar order.
2. Identifies and patches the single omitted calendar date (Sunday, 2016-09-18)
   using power-system-standard Weekly Seasonal Linear Interpolation (Sunday-to-Sunday):
       v(t) = v_prev + ((v_next - v_prev) / (t_next - t_prev)) * (t - t_prev)
   where:
       t_prev = 2016-09-11 (Sunday, t - 7 days)
       t      = 2016-09-18 (Sunday, missing date)
       t_next = 2016-09-25 (Sunday, t + 7 days)
       delta_t_step  = 7 days
       delta_t_total = 14 days
       alpha = 7 / 14 = 0.5
   This preserves weekly cyclical load seasonality (avoiding weekday commercial/industrial
   bleed into weekend residential load) and preserves informative missingness (NaN remains NaN).
3. Formats all 1,035 daily date column headers to standard ISO-8601 (YYYY-MM-DD).
4. Emits verified continuous time series into data/SGCC.csv (147 weeks x 7 days).

Idempotent: If data/SGCC.csv is already sorted and complete, verifies and exits.
"""
from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO_ROOT = THIS.parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from repro_config import DATA_DIR, SGCC_PATH

DEFAULT_RAW_CANDIDATES = [
    DATA_DIR / "SGCC_raw.csv",
    DATA_DIR / "SGCC.csv",
]


def is_already_processed(csv_path: Path) -> bool:
    """Fast check to verify if the target CSV is already chronologically ordered and complete."""
    if not csv_path.exists():
        return False
    try:
        import pandas as pd
        hdr = pd.read_csv(csv_path, nrows=2)
        if len(hdr.columns) != 1037:  # CONS_NO, FLAG + 1035 dates
            return False
        date_cols = list(hdr.columns[2:])
        if "2016-09-18" not in date_cols:
            return False
        parsed = pd.to_datetime(date_cols, format="%Y-%m-%d")
        diffs = pd.Series(parsed).diff().dropna().unique()
        return len(diffs) == 1 and diffs[0] == pd.Timedelta(days=1)
    except Exception:
        return False


def prepare_dataset(force: bool = False) -> None:
    target_path = SGCC_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SGCC DATASET CHRONOLOGICAL PREPARATION & AUDIT")
    print(f"Target Destination: {target_path}")
    print("=" * 75)

    if not force and is_already_processed(target_path):
        print(f"[OK] Target file {target_path} is already chronologically ordered,")
        print("     fully continuous (1,035 days: 2014-01-01 -> 2016-10-31),")
        print("     and 2016-09-18 is present. Skipping re-generation.")
        print("=" * 75 + "\n")
        return

    import numpy as np
    import pandas as pd

    # Locate source file
    source_path = None
    env_raw = os.environ.get("MAMRD_RAW_SGCC")
    if env_raw and Path(env_raw).exists():
        source_path = Path(env_raw)
    else:
        for cand in DEFAULT_RAW_CANDIDATES:
            if cand.exists() and cand.resolve() != target_path.resolve():
                source_path = cand
                break
        if source_path is None and target_path.exists():
            source_path = target_path

    if source_path is None or not source_path.exists():
        raise FileNotFoundError(
            f"Could not locate source SGCC raw dataset. Checked: {DEFAULT_RAW_CANDIDATES}"
        )

    print(f">> Ingesting source SGCC dataset from: {source_path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(source_path)
    print(f"   Loaded shape: {df.shape} in {time.time()-t0:.2f}s")

    meta_cols = ["CONS_NO", "FLAG"]
    date_cols = [c for c in df.columns if c not in meta_cols]
    parsed_map = {c: pd.to_datetime(c, format="mixed") for c in date_cols}

    expected_range = pd.date_range("2014-01-01", "2016-10-31", freq="D")
    all_parsed_set = set(parsed_map.values())
    missing_dates = sorted(set(expected_range) - all_parsed_set)
    print(f"   Identified missing calendar dates: {[d.strftime('%Y-%m-%d') for d in missing_dates]}")

    # Patch 2016-09-18 (Sunday) via Weekly Seasonal Linear Interpolation (Sunday-to-Sunday)
    patch_dict = {}
    if pd.Timestamp("2016-09-18") in missing_dates:
        t_prev   = pd.Timestamp("2016-09-11")  # Sunday (t - 7 days)
        t_target = pd.Timestamp("2016-09-18")  # Sunday (target missing day)
        t_next   = pd.Timestamp("2016-09-25")  # Sunday (t + 7 days)

        col_prev = [c for c, dt in parsed_map.items() if dt == t_prev][0]
        col_next = [c for c, dt in parsed_map.items() if dt == t_next][0]
        v_prev = df[col_prev].to_numpy(dtype=np.float64)
        v_next = df[col_next].to_numpy(dtype=np.float64)

        # Standard linear interpolation along the weekly seasonal axis:
        # v(t) = v_prev + ((v_next - v_prev) / (t_next - t_prev)) * (t - t_prev)
        dt_total = (t_next - t_prev).total_seconds()  # 14 days in seconds
        dt_step  = (t_target - t_prev).total_seconds() # 7 days in seconds
        v_seasonal_linear = v_prev + ((v_next - v_prev) / dt_total) * dt_step

        # Informative missingness firewall: natural missingness is strictly preserved
        v18 = np.where(
            np.isnan(v_prev) & np.isnan(v_next),
            np.nan,
            np.where(np.isnan(v_prev), v_next, np.where(np.isnan(v_next), v_prev, v_seasonal_linear))
        )
        patch_dict[t_target] = v18
        print(f"   Successfully synthesized Sunday 2016-09-18 via Weekly Seasonality Linear Interpolation:")
        print(f"   v(t) = v_prev + ((v_next - v_prev) / (t_next - t_prev)) * (t - t_prev)")
        print(f"   Anchor Sundays: {t_prev.strftime('%Y-%m-%d (%A)')} and {t_next.strftime('%Y-%m-%d (%A)')}")

    # Build chronologically ordered DataFrame with standard ISO headers
    ts_to_col = {dt: c for c, dt in parsed_map.items()}
    out_dict = {"CONS_NO": df["CONS_NO"], "FLAG": df["FLAG"]}

    for dt in expected_range:
        iso_str = dt.strftime("%Y-%m-%d")
        if dt in patch_dict:
            out_dict[iso_str] = patch_dict[dt]
        else:
            orig_col = ts_to_col[dt]
            out_dict[iso_str] = df[orig_col].to_numpy()

    df_out = pd.DataFrame(out_dict)
    print(f">> Writing chronologically aligned CSV to {target_path} ...", flush=True)
    t1 = time.time()
    df_out.to_csv(target_path, index=False)
    print(f"   Saved {df_out.shape} in {time.time()-t1:.2f}s.")

    # Integrity verification
    assert len(df_out.columns) == 1037, f"Expected 1037 columns, got {len(df_out.columns)}"
    assert "2016-09-18" in df_out.columns, "2016-09-18 missing in generated file"
    print(f"[PASS] Preprocessed dataset verified successfully (100% chronological continuity).")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare chronologically sorted SGCC dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if valid.")
    args = parser.parse_args()
    prepare_dataset(force=args.force)
