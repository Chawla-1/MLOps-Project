"""Detect the 7 data-quality issues in train.csv and write the data_issues
section of reports/findings.yaml. Every value comes from the data itself.

Usage:
    python -m src.inspect_data --train data/raw/train.csv \
        --holdout data/raw/holdout_features.csv --findings reports/findings.yaml
"""
import argparse
import re

import numpy as np
import pandas as pd
import yaml

TARGET = "rent_rs"
NUMERIC = ["distance_to_campus_km", "room_area_sqft", "sharing",
           "meals_per_day", "building_age_yr", "wifi_mbps"]
CATEGORICAL = ["furnishing", "locality"]
COMMON_FACTORS = [0.0929, 0.3048, 0.001, 0.01, 0.1, 0.62137, 0.9144,
                  10.0, 100.0, 1000.0, 10.764, 3.28084, 1.60934, 0.3937,
                  2.54, 0.4536, 2.20462]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r"[\s_\-\.]+", "", regex=True)


def detect_leaking(train, holdout):
    extra = [c for c in train.columns
             if c not in holdout.columns and c not in (TARGET,)]
    if not extra:  # fallback: suspiciously perfect correlation with target
        num = train.select_dtypes("number").drop(columns=[TARGET], errors="ignore")
        corr = num.corrwith(train[TARGET]).abs()
        extra = [c for c in corr.index if corr[c] > 0.98]
    return {"present": bool(extra), "column": extra[0] if extra else None}


def detect_unit_mixup(train):
    best = None
    for col in NUMERIC:
        if col not in train or "locality" not in train:
            continue
        med = train.groupby("locality")[col].median()
        for g in med.index:
            others = train.loc[train["locality"] != g, col].median()
            if others and med[g] > 0:
                ratio = med[g] / others
                dev = abs(np.log(ratio))
                if best is None or dev > best[0]:
                    best = (dev, col, g, ratio)
    if best and best[0] > np.log(2.0):
        _, col, g, ratio = best
        factor = min(COMMON_FACTORS, key=lambda f: abs(np.log(f / ratio)))
        if abs(np.log(factor / ratio)) > 0.25:
            factor = round(float(ratio), 4)
        return {"present": True, "column": col, "group_value": g, "factor": float(factor)}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def detect_sentinel(train):
    best = None
    for col in NUMERIC:
        if col not in train:
            continue
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        vc = s.value_counts()
        for v, c in vc.items():
            looks_sentinel = (v in (-999, -99, -9, -1, 999, 9999, 99999, 0) or
                              (abs(v) >= 99 and set(str(abs(int(v)))) == {"9"}))
            far = v < s.quantile(0.01) - 3 * s.std() or v > s.quantile(0.99) + 3 * s.std()
            if c >= 5 and (looks_sentinel and (far or v < 0 or v >= 99)):
                if best is None or c > best[3]:
                    best = (col, v, None, int(c))
    if best:
        v = best[1]
        return {"present": True, "column": best[0],
                "value": int(v) if float(v).is_integer() else float(v), "count": best[3]}
    return {"present": False, "column": None, "value": None, "count": None}


def detect_duplicates(train):
    full = int(train.duplicated().sum())
    no_id = int(train.drop(columns=["record_id"], errors="ignore").duplicated().sum())
    count = full if full else no_id
    return {"present": count > 0, "count": count or None}


def detect_stuck(train, min_run=15):
    if "date" not in train:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    d = train.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "record_id"], kind="stable")
    best = None
    for col in NUMERIC:
        s = d[col].reset_index(drop=True)
        dates = d["date"].reset_index(drop=True)
        run_id = (s != s.shift()).cumsum()
        for _, idx in s.groupby(run_id).groups.items():
            n = len(idx)
            if n >= min_run and (best is None or n > best[0]):
                best = (n, col, dates[idx[0]], dates[idx[-1]])
    if best:
        return {"present": True, "column": best[1],
                "start_date": best[2].strftime("%Y-%m-%d"),
                "end_date": best[3].strftime("%Y-%m-%d")}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def detect_target_typos(train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    df = train.dropna(subset=[TARGET]).copy()
    X = df[[c for c in NUMERIC if c in df]].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    for c in CATEGORICAL:
        if c in df:
            X[c] = _norm(df[c]).astype("category").cat.codes
    med = df[TARGET].median()
    rough = df[TARGET] < 4 * med           # train on the obviously-sane rows
    if rough.all():                        # nothing anywhere near 10x the median
        return {"present": False, "count": None}
    pred = pd.Series(np.nan, index=df.index)
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    pred[rough] = cross_val_predict(rf, X[rough], df.loc[rough, TARGET], cv=5)
    rf.fit(X[rough], df.loc[rough, TARGET])
    pred[~rough] = rf.predict(X[~rough])
    ratio = df[TARGET] / pred
    count = int((ratio > 5).sum())
    return {"present": count > 0, "count": count or None}


def detect_spelling(train, expected=None):
    from src.data_cleaning import canonical_map
    for col in CATEGORICAL:
        raw = train[col].dropna().astype(str).nunique()
        true = canonical_map(train[col].dropna()).nunique()
        if raw > true:
            return {"present": True, "column": col, "true_levels": int(true)}
    return {"present": False, "column": None, "true_levels": None}


def run(train_path, holdout_path):
    train = pd.read_csv(train_path)
    holdout = pd.read_csv(holdout_path)
    return {
        "leaking_column": detect_leaking(train, holdout),
        "unit_mixup": detect_unit_mixup(train),
        "sentinel_value": detect_sentinel(train),
        "duplicate_rows": detect_duplicates(train),
        "stuck_sensor": detect_stuck(train),
        "target_typos": detect_target_typos(train),
        "spelling_variants": detect_spelling(train),
    }


def write_findings(path, data_issues):
    """Replace only the data_issues block, keeping comments elsewhere intact."""
    text = open(path).read()
    block = yaml.safe_dump({"data_issues": data_issues}, sort_keys=False, default_flow_style=False)
    new = re.sub(r"data_issues:.*?(?=\ntracking:)", block.rstrip() + "\n", text, flags=re.S)
    open(path, "w").write(new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--holdout", default="data/raw/holdout_features.csv")
    ap.add_argument("--findings", default="reports/findings.yaml")
    ap.add_argument("--write", action="store_true", help="write into findings.yaml")
    a = ap.parse_args()
    res = run(a.train, a.holdout)
    print(yaml.safe_dump({"data_issues": res}, sort_keys=False))
    if a.write:
        write_findings(a.findings, res)
        print("findings.yaml updated")
