import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os
import streamlit as st


class MLProcessor:
    """Base class for machine learning operations"""

    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.results = {}

    def prepare_data(self, df, target_column, feature_columns=None, test_size=0.2, random_state=42):
        """
        Prepare data for model training by splitting into train and test sets

        Args:
            df: pandas DataFrame containing the data
            target_column: name of the target column
            feature_columns: list of feature column names (if None, use all columns except target)
            test_size: proportion of data to use for testing (default 0.2)
            random_state: random state for reproducibility

        Returns:
            dict: containing X_train, X_test, y_train, y_test, and feature_names
        """
        # If feature columns not specified, use all columns except target
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        # Ensure all feature columns exist in the dataframe
        usable_features = [col for col in feature_columns if col in df.columns]

        if not usable_features:
            raise ValueError("No valid features available for model training.")

        # Get features (X) and target (y)
        X = df[usable_features].copy()
        y = df[target_column].copy()

        # Drop rows with NaN values in either X or y
        valid_rows = ~(X.isna().any(axis=1) | y.isna())
        if valid_rows.sum() == 0:
            raise ValueError("No valid rows after removing missing values.")

        X = X[valid_rows]
        y = y[valid_rows]

        # Check if target is numeric for regression
        if not pd.api.types.is_numeric_dtype(y):
            if isinstance(self, type) and self.__name__ == 'RegressionProcessor':
                raise ValueError("Target column must be numeric for regression tasks.")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': usable_features
        }

    def register_model(self, model_name, model_instance, model_config=None):
        """
        Register a model for use

        Args:
            model_name: name of the model
            model_instance: instance of the model
            model_config: configuration parameters for the model
        """
        self.models[model_name] = model_instance
        self.model_configs[model_name] = model_config or {}

    def train_model(self, model_name, X_train, y_train):
        """
        Train a registered model

        Args:
            model_name: name of the registered model
            X_train: training features
            y_train: training target

        Returns:
            trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model

        Args:
            model_name: name of the trained model
            X_test: test features
            y_test: test target

        Returns:
            dict: evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        # This method should be implemented by subclasses for specific types of models
        raise NotImplementedError("Subclasses must implement evaluate_model")

    def cross_validate(self, model_name, X, y, cv=5, scoring=None):
        """
        Perform cross-validation on a model

        Args:
            model_name: name of the registered model
            X: features
            y: target
            cv: number of cross-validation folds
            scoring: scoring metric

        Returns:
            numpy array: cross-validation scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores

    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk

        Args:
            model_name: name of the trained model
            filepath: path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        model = self.models[model_name]
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_name, filepath):
        """
        Load a model from disk

        Args:
            model_name: name to register the loaded model
            filepath: path to the saved model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        self.register_model(model_name, model)