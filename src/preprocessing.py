"""
Data Preprocessing Utilities for Work 3 - IML Clustering.

Handles:
- Loading ARFF files using the existing parser.
- Separating features X and ground-truth labels y.
- Missing value imputation:
    * Median imputation for numerical features (robust against outliers).
    * Most frequent (mode) imputation for categorical features.
- Categorical encoding: One-hot encoding for nominal attributes.
- Feature scaling: StandardScaler for continuous variables.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
try:
    from src.parser import load_arff
except ModuleNotFoundError:
    from parser import load_arff


def preprocess_kr_vs_kp(file_path="data/kr-vs-kp.arff"):
    """
    Preprocess the King-Rook vs King-Pawn (kr-vs-kp) dataset.
    
    All 36 attributes are nominal (categorical) with zero missing values.
    We apply one-hot encoding across all categorical attributes to map them
    into a numeric coordinate space suitable for distance-based clustering.
    
    Returns:
        X_encoded (np.ndarray): Preprocessed feature matrix (samples x features).
        y (np.ndarray): Ground-truth class labels ('won' / 'nowin').
        feature_names (list): List of transformed feature names.
    """
    X, y = load_arff(file_path)
    
    # All features are categorical; one-hot encode them
    X_encoded_df = pd.get_dummies(X, drop_first=False, dtype=float)
    
    return X_encoded_df.values, y.values, list(X_encoded_df.columns)


def preprocess_horse_colic(file_path="data/horse-colic.arff"):
    """
    Preprocess the Horse-Colic dataset.
    
    This dataset has mixed attributes (7 numeric, 15 categorical) and contains
    missing values.
    
    Steps:
      1. Separate numeric and categorical features.
      2. Impute numeric missing values using median.
      3. Standardize numeric features (zero mean, unit variance) so they do
         not dominate the distance metric compared to one-hot binary features.
      4. Impute categorical missing values using most frequent (mode).
      5. One-hot encode categorical features.
      6. Concatenate numeric and encoded categorical features.
    
    Returns:
        X_processed (np.ndarray): Preprocessed numeric matrix.
        y (np.ndarray): Ground-truth labels (surgical_lesion: 'yes' / 'no').
        feature_names (list): List of transformed feature names.
    """
    X, y = load_arff(file_path)
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 1. Handle numeric attributes
    num_imputer = SimpleImputer(strategy="median")
    X_num_imputed = num_imputer.fit_transform(X[num_cols])
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)
    df_num = pd.DataFrame(X_num_scaled, columns=num_cols)
    
    # 2. Handle categorical attributes
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_cat_imputed = cat_imputer.fit_transform(X[cat_cols])
    df_cat = pd.DataFrame(X_cat_imputed, columns=cat_cols)
    df_cat_encoded = pd.get_dummies(df_cat, drop_first=False, dtype=float)
    
    # 3. Concatenate processed columns
    df_processed = pd.concat([df_num, df_cat_encoded], axis=1)
    
    return df_processed.values, y.values, list(df_processed.columns)


def load_dataset(name, data_dir="data"):
    """
    Convenience loader for a named dataset.
    
    Supported names: 'kr-vs-kp', 'horse-colic'.
    """
    if name == "kr-vs-kp":
        return preprocess_kr_vs_kp(f"{data_dir}/kr-vs-kp.arff")
    elif name == "horse-colic":
        return preprocess_horse_colic(f"{data_dir}/horse-colic.arff")
    else:
        raise ValueError(f"Unknown dataset name: {name}. Expected 'kr-vs-kp' or 'horse-colic'.")


if __name__ == "__main__":
    for ds in ["kr-vs-kp", "horse-colic"]:
        X, y, feats = load_dataset(ds)
        print(f"Loaded {ds}: shape={X.shape}, labels={len(np.unique(y))}, NaNs={np.isnan(X).sum()}")
