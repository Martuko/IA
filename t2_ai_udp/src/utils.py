
from __future__ import annotations
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    csv_path = cfg["dataset_path"]
    target_col = cfg["target_column"]
    drop_cols = cfg.get("drop_columns", []) or []
    cat_cols = cfg.get("categorical_columns", []) or []
    df = pd.read_csv(csv_path)
    # Drop specified columns if present
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Basic sanity checks
    assert target_col in df.columns, f"La columna objetivo '{target_col}' no se encuentra en el dataset."
    # Figure out numeric/categorical columns
    # If user specified cat_cols, use them; else infer object/category as categorical
    if not cat_cols:
        inferred_cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if target_col in inferred_cat:
            inferred_cat.remove(target_col)
        cat_cols = inferred_cat
    num_cols = [c for c in df.columns if c != target_col and c not in cat_cols]
    return df, target_col, num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = StandardScaler(with_mean=False)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=True, drop=None)
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # keep sparse when appropriate
    )
    return pre

def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def make_splits(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
