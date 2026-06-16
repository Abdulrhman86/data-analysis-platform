# utils/feature_selection.py
import pandas as pd
import numpy as np
from utils.logging_setup import get_logger
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

logger = get_logger(__name__)
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def filter_numeric_features(X, feature_names=None):
    """
    Filter out non-numeric columns from a feature matrix

    Args:
        X: feature matrix (DataFrame)
        feature_names: list of feature names (if None, use X.columns)

    Returns:
        tuple: (filtered DataFrame, list of numeric feature names)
    """
    if feature_names is None:
        feature_names = X.columns.tolist()

    # Check each column for numeric type
    numeric_cols = []
    for col in feature_names:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            numeric_cols.append(col)

    return X[numeric_cols], numeric_cols


def filter_valid_data(X, y):
    """
    Filter out rows with missing values in either X or y

    Args:
        X: feature matrix
        y: target vector

    Returns:
        tuple: (filtered X, filtered y)
    """
    # Combine X and y to identify rows with any missing values
    combined = pd.concat([X, pd.Series(y, name='target')], axis=1)

    # Drop rows with any missing values
    valid_data = combined.dropna()

    if len(valid_data) == 0:
        raise ValueError("No complete rows found after removing missing values. Please handle missing values first.")

    if len(valid_data) < len(X):
        logger.info("Feature selection dropped %d rows with missing values.", len(X) - len(valid_data))

    # Split back into X and y
    X_filtered = valid_data.drop('target', axis=1)
    y_filtered = valid_data['target']

    return X_filtered, y_filtered


def select_k_best_features(X, y, k=10, task='classification'):
    """
    Select k best features using statistical tests
    Args:
        X: feature matrix
        y: target vector
        k: number of features to select
        task: 'classification' or 'regression'
    Returns:
        list: selected feature names
    """
    # For regression, use only numeric columns
    # For classification, keep all columns (including categorical ones)
    if task == 'regression':
        # Filter out non-numeric columns for regression tasks
        usable_cols = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
                usable_cols.append(col)
        if not usable_cols:
            raise ValueError("No numeric features available for regression.")
    else:
        # Keep all columns for classification tasks
        usable_cols = X.columns.tolist()
        if not usable_cols:
            raise ValueError("No features available for selection.")

    # Use selected columns
    X_usable = X[usable_cols]

    # Remove rows with missing values
    valid_mask = ~(X_usable.isna().any(axis=1) | y.isna())
    X_filtered = X_usable[valid_mask]
    y_filtered = y[valid_mask]

    if len(X_filtered) == 0:
        raise ValueError("No valid rows after removing missing values.")

    # Choose scoring function based on task
    if task == 'classification':
        score_func = f_classif
    else:
        score_func = f_regression

    # Limit k to the number of available features
    k = min(k, len(usable_cols))

    # Create selector
    selector = SelectKBest(score_func=score_func, k=k)

    try:
        # Fit the selector
        selector.fit(X_filtered, y_filtered)

        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)

        # Convert indices to feature names
        selected_features = [usable_cols[i] for i in selected_indices]

        return selected_features[:k]  # Explicitly limit to k
    except Exception as e:
        raise ValueError(f"Error during feature selection: {str(e)}")

def select_features_with_rfe(X, y, n_features=10, task='classification', step=1):
    """
    Select features using Recursive Feature Elimination

    Args:
        X: feature matrix
        y: target vector
        n_features: number of features to select
        task: 'classification' or 'regression'
        step: number of features to remove at each iteration

    Returns:
        selected feature names
    """
    # Choose estimator based on task
    if task == 'classification':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    # Create selector
    selector = RFE(estimator, n_features_to_select=n_features, step=step)
    selector.fit(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get feature names
    feature_names = X.columns[selected_indices]

    return list(feature_names)


def select_features_with_mutual_info(X, y, k=10, task='classification'):
    """
    Select features using mutual information

    Args:
        X: feature matrix
        y: target vector
        k: number of features to select
        task: 'classification' or 'regression'

    Returns:
        tuple: (selected feature names, scores series)
    """
    # Filter out non-numeric columns first
    numeric_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric features available for selection.")

    # Use only numeric columns
    X_numeric = X[numeric_cols]

    # Remove rows with missing values
    valid_mask = ~(X_numeric.isna().any(axis=1) | y.isna())
    X_filtered = X_numeric[valid_mask]
    y_filtered = y[valid_mask]

    if len(X_filtered) == 0:
        raise ValueError("No valid rows after removing missing values.")

    # Limit k to the number of available features
    k = min(k, len(numeric_cols))

    # Choose scoring function based on task
    if task == 'classification':
        score_func = mutual_info_classif
    else:
        score_func = mutual_info_regression

    try:
        # Calculate mutual information
        mi_scores = score_func(X_filtered, y_filtered, random_state=42)

        # Create Series of scores
        scores = pd.Series(mi_scores, index=numeric_cols)

        # Sort scores and select top k
        sorted_scores = scores.sort_values(ascending=False)
        selected_features = sorted_scores.head(k).index.tolist()

        return selected_features, sorted_scores

    except Exception as e:
        raise ValueError(f"Error during feature selection: {str(e)}")


def get_feature_importance_from_model(model, feature_names):
    """
    Get feature importance from a trained model

    Args:
        model: trained model
        feature_names: list of feature names

    Returns:
        pandas Series of feature importance
    """
    # Check if model has feature importance attribute
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, 'coef_'):
        # For linear models with coefficients
        if len(model.coef_.shape) > 1:
            # For multi-class, take the mean absolute coefficient
            importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            importance = np.abs(model.coef_)
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
    else:
        return None