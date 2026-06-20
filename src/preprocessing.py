"""
preprocessing.py
Impute, split, and scale the credit dataset.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame):
    """
    Impute missing values, split features/target, scale numerics.
    Returns: X_train, X_test, y_train, y_test, feature_names, imputer, scaler
    """
    target       = "creditworthy"
    X            = df.drop(columns=[target])
    y            = df[target]
    feature_names= X.columns.tolist()

    imputer   = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    return X_train_sc, X_test_sc, X_train, X_test, y_train, y_test, feature_names, imputer, scaler
