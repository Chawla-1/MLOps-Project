"""Reusable cleaning functions for MP141 train.csv. Used by train.py and monitor.py."""
import numpy as np
import pandas as pd
import yaml

TARGET = "rent_rs"
NUMERIC = ["distance_to_campus_km", "room_area_sqft", "sharing",
           "meals_per_day", "building_age_yr", "wifi_mbps"]


def drop_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def _key(s):
    """lower-case, drop spaces/dots/dashes: ' Unfurnished', 'unf.' -> 'unfurnished', 'unf'"""
    return s.astype(str).str.strip().str.lower().str.replace(r"[\s_\-\.]+", "", regex=True)


def canonical_map(series):
    """Map every spelling to one clean level.
    A key that is only a short form of a longer key ('unf' of 'unfurnished')
    is merged into the longer one. The clean label is the most common full spelling."""
    keys = _key(series)
    uniq = sorted(keys.unique(), key=len, reverse=True)
    full = [k for k in uniq if not any(o != k and o.startswith(k) for o in uniq)]
    to_full = {k: next(f for f in full if f.startswith(k)) for k in uniq}
    return keys.map(to_full)


def fix_spelling(df, col):
    df = df.copy()
    df[col] = canonical_map(df[col])
    return df


def fix_unit(df, col, group_value, factor):
    df = df.copy()
    m = df["locality"] == group_value
    df.loc[m, col] = df.loc[m, col] / factor
    return df


def fix_sentinel(df, col, value):
    df = df.copy()
    df.loc[df[col] == value, col] = np.nan
    return df


def blank_stuck(df, col, start, end):
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df.loc[(d >= start) & (d <= end), col] = np.nan
    return df


def fix_target_typos(df, ratio_mask=None, factor=10.0):
    """Divide typo'd targets by 10. Pass a boolean mask or let it use a median rule."""
    df = df.copy()
    if ratio_mask is None:
        ratio_mask = df[TARGET] > 4 * df[TARGET].median()
    df.loc[ratio_mask, TARGET] = df.loc[ratio_mask, TARGET] / factor
    return df


def impute_numeric(df, medians=None):
    df = df.copy()
    medians = medians or df[NUMERIC].median().to_dict()
    for c in NUMERIC:
        df[c] = df[c].fillna(medians[c])
    return df, medians


def clean_train(df, findings_path="reports/findings.yaml", holdout_cols=None):
    """Apply every fix described in findings.yaml -> data_issues."""
    di = yaml.safe_load(open(findings_path))["data_issues"]
    if di["duplicate_rows"]["present"]:
        df = drop_duplicates(df)
    if di["spelling_variants"]["present"]:
        df = fix_spelling(df, di["spelling_variants"]["column"])
    if di["unit_mixup"]["present"]:
        u = di["unit_mixup"]
        df = fix_unit(df, u["column"], u["group_value"], u["factor"])
    if di["sentinel_value"]["present"]:
        s = di["sentinel_value"]
        df = fix_sentinel(df, s["column"], s["value"])
    if di["stuck_sensor"]["present"]:
        k = di["stuck_sensor"]
        df = blank_stuck(df, k["column"], k["start_date"], k["end_date"])
    if di["target_typos"]["present"]:
        df = fix_target_typos(df)
    if di["leaking_column"]["present"]:
        df = df.drop(columns=[di["leaking_column"]["column"]])
    df, _ = impute_numeric(df)
    return df
