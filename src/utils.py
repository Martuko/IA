from __future__ import annotations
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _derive_failure_type_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el CSV no trae 'Failure Type' pero sí las one-hot TWF/HDF/PWF/OSF/RNF,
    crea la etiqueta multiclase:
      - TWF -> 'Tool Wear Failure'
      - HDF -> 'Heat Dissipation Failure'
      - PWF -> 'Power Failure'
      - OSF -> 'Overstrain Failure'
      - RNF -> 'Random Failures'
      - Si todas son 0 -> 'No Failure'
    """
    ft_col = "Failure Type"
    onehots = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    if ft_col not in df.columns and all(c in df.columns for c in onehots):
        mapping = {
            "TWF": "Tool Wear Failure",
            "HDF": "Heat Dissipation Failure",
            "PWF": "Power Failure",
            "OSF": "Overstrain Failure",
            "RNF": "Random Failures",
        }

        def row_to_label(row):
            for c in onehots:
                try:
                    val = int(row[c])  # soporta 0/1 como float/str
                except Exception:
                    val = row[c]
                if val == 1:
                    return mapping[c]
            return "No Failure"

        df[ft_col] = df.apply(row_to_label, axis=1)
    return df


def load_dataset(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    csv_path = cfg["dataset_path"]
    target_col = cfg["target_column"]
    drop_cols = cfg.get("drop_columns", []) or []
    cat_cols = cfg.get("categorical_columns", []) or []

    # 1) Leer CSV (strip en nombres por si vienen con espacios accidentales)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 2) Derivar 'Failure Type' si no existe
    df = _derive_failure_type_if_needed(df)

    # 3) Eliminar columnas indicadas (evita fuga de etiquetas)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 4) Chequeo básico
    assert target_col in df.columns, f"La columna objetivo '{target_col}' no se encuentra en el dataset."

    # 5) Detectar columnas categóricas / numéricas
    if not cat_cols:
        inferred_cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if target_col in inferred_cat:
            inferred_cat.remove(target_col)
        cat_cols = inferred_cat

    num_cols = [c for c in df.columns if c != target_col and c not in cat_cols]

    return df, target_col, num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    # StandardScaler: usar with_mean=False por compatibilidad con salidas dispersas
    numeric_transformer = StandardScaler(with_mean=False)

    # En scikit-learn 1.7+ se usa 'sparse_output' (NO 'sparse')
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        drop=None
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # puede devolver sparse si conviene
    )
    return pre


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def make_splits(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
