from pathlib import Path
from typing import Any

import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_manifest(split_path: str | Path) -> dict[str, Any]:
    split_path = Path(split_path)

    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_units(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    csv_path = data_root / "units_public.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    return pd.read_csv(csv_path)


def split_by_unit_ids(df: pd.DataFrame, manifest: dict[str, Any],) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_ids = manifest["splits"]["train"]
    val_ids = manifest["splits"]["val"]
    test_ids = manifest["splits"]["test"]

    train_df = df[df["unit_id"].isin(train_ids)].copy()
    val_df = df[df["unit_id"].isin(val_ids)].copy()
    test_df = df[df["unit_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def apply_rul_cap(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    df = df.copy()

    if "rul" in df.columns:
        df["rul_capped"] = df["rul"].clip(upper=rul_cap)

    return df


def fit_train_scaler(train_df: pd.DataFrame,feature_columns: list[str],) -> StandardScaler:

    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
    return scaler


def transform_features(df: pd.DataFrame,feature_columns: list[str],scaler: StandardScaler,) -> pd.DataFrame:
    df = df.copy()
    df[feature_columns] = scaler.transform(df[feature_columns])
    return df


def prepare_supervised_splits(data_root: str | Path,split_path: str | Path,rul_cap: int,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], StandardScaler]:
    
    manifest = load_manifest(split_path)
    df = load_units(data_root)

    feature_columns = manifest["feature_columns"]

    train_df, val_df, test_df = split_by_unit_ids(df, manifest)

    train_df = apply_rul_cap(train_df, rul_cap)
    val_df = apply_rul_cap(val_df, rul_cap)
    test_df = apply_rul_cap(test_df, rul_cap)

    scaler = fit_train_scaler(train_df, feature_columns)

    train_df = transform_features(train_df, feature_columns, scaler)
    val_df = transform_features(val_df, feature_columns, scaler)
    test_df = transform_features(test_df, feature_columns, scaler)

    return train_df, val_df, test_df, feature_columns, scaler