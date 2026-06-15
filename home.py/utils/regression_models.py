# utils/regression_models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.ml_processor import MLProcessor


class RegressionProcessor(MLProcessor):
    """Class for regression model processing"""

    def __init__(self):
        super().__init__()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize and register standard regression models"""
        # Linear Regression
        self.register_model(
            "Linear Regression",
            LinearRegression(),
            {}
        )

        # Ridge Regression
        self.register_model(
            "Ridge Regression",
            Ridge(random_state=42),
            {
                "alpha": 1.0
            }
        )

        # Lasso Regression
        self.register_model(
            "Lasso Regression",
            Lasso(random_state=42),
            {
                "alpha": 1.0
            }
        )

        # Decision Tree Regressor
        self.register_model(
            "Decision Tree Regressor",
            DecisionTreeRegressor(random_state=42),
            {
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        )

        # Random Forest Regressor
        self.register_model(
            "Random Forest Regressor",
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        )

    def update_model_parameters(self, model_name, parameters):
        """
        Update model parameters

        Args:
            model_name: name of the registered model
            parameters: new parameters for the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        # Create new model instance with updated parameters
        if model_name == "Linear Regression":
            model = LinearRegression(**parameters)
        elif model_name == "Ridge Regression":
            model = Ridge(random_state=42, **parameters)
        elif model_name == "Lasso Regression":
            model = Lasso(random_state=42, **parameters)
        elif model_name == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42, **parameters)
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(random_state=42, **parameters)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Update model registry
        self.register_model(model_name, model, parameters)

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a regression model

        Args:
            model_name: name of the registered model
            X_test: test features
            y_test: test target

        Returns:
            dict: evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        model = self.models[model_name]

        # Make predictions
        y_pred = model.predict(X_test)

        # Convert datetime target to numeric if needed
        if pd.api.types.is_datetime64_dtype(y_test):
            # Convert to days since epoch or another reference date
            if isinstance(y_test, pd.Series):
                numeric_y_test = y_test.map(lambda x: x.timestamp())
            else:
                numeric_y_test = np.array([x.timestamp() for x in y_test])
        else:
            numeric_y_test = y_test

        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(numeric_y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(numeric_y_test, y_pred)),
            'mae': mean_absolute_error(numeric_y_test, y_pred),
            'r2': r2_score(numeric_y_test, y_pred)
        }

        # Store results
        self.results[model_name] = metrics

        return metrics

    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for a model (if available)

        Args:
            model_name: name of the registered model
            feature_names: list of feature names

        Returns:
            pd.Series: feature importance values
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        model = self.models[model_name]

        # Check if model has feature importance attribute
        if hasattr(model, 'feature_importances_'):
            return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            # For linear models with coefficients
            return pd.Series(np.abs(model.coef_), index=feature_names).sort_values(ascending=False)
        else:
            return None