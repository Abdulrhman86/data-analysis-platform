# utils/classification_models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.ml_processor import MLProcessor


class ClassificationProcessor(MLProcessor):
    """Class for classification model processing"""
    task_type = 'classification'

    def __init__(self):
        super().__init__()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize and register standard classification models"""
        # Logistic Regression
        self.register_model(
            "Logistic Regression",
            LogisticRegression(random_state=42),
            {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100
            }
        )

        # Decision Tree
        self.register_model(
            "Decision Tree",
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        )

        # Random Forest
        self.register_model(
            "Random Forest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        )

        # Support Vector Machine
        self.register_model(
            "SVM",
            SVC(probability=True, random_state=42),
            {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"
            }
        )

        # Naive Bayes
        self.register_model(
            "Naive Bayes",
            GaussianNB(),
            {}
        )

        # K-Nearest Neighbors (benefits from the pipeline's feature scaling)
        self.register_model(
            "K-Nearest Neighbors",
            KNeighborsClassifier(),
            {"n_neighbors": 5}
        )

        # Gradient Boosting (histogram-based; strong non-linear baseline)
        self.register_model(
            "Gradient Boosting",
            HistGradientBoostingClassifier(random_state=42),
            {"learning_rate": 0.1, "max_depth": None}
        )

        # Default to balanced class weights for models that support it, so an
        # imbalanced target doesn't collapse to predicting the majority class.
        # (KNN and Naive Bayes have no class_weight and are left untouched.)
        for model in self.models.values():
            if "class_weight" in model.get_params():
                model.set_params(class_weight="balanced")

    def get_param_grid(self, model_name):
        """Small hyperparameter grids used by tune_model."""
        grids = {
            "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
            "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5]},
            "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
            "SVM": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 11]},
            "Gradient Boosting": {"learning_rate": [0.05, 0.1], "max_depth": [None, 3, 5]},
        }
        return grids.get(model_name, {})

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
        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, **parameters)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42, **parameters)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42, **parameters)
        elif model_name == "SVM":
            model = SVC(probability=True, random_state=42, **parameters)
        elif model_name == "Naive Bayes":
            model = GaussianNB(**parameters)
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier(**parameters)
        elif model_name == "Gradient Boosting":
            model = HistGradientBoostingClassifier(random_state=42, **parameters)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Keep balanced class weights when the model supports it.
        if "class_weight" in model.get_params():
            model.set_params(class_weight="balanced")

        # Update model registry
        self.register_model(model_name, model, parameters)

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a classification model

        Args:
            model_name: name of the registered model
            X_test: test features
            y_test: test target

        Returns:
            dict: evaluation metrics
        """
        model = self.get_fitted(model_name)

        # Make predictions
        y_pred = model.predict(X_test)

        # For ROC AUC, we need probability predictions
        try:
            y_proba = model.predict_proba(X_test)
            classes = model.classes_
            # For multi-class, use one-vs-rest with an explicit class order so the
            # probability columns line up with the true labels.
            if y_proba.shape[1] > 2:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', labels=classes)
            else:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except Exception:
            roc_auc = None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Store results
        self.results[model_name] = metrics

        return metrics

    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for a trained model (if available)

        Args:
            model_name: name of the trained model
            feature_names: fallback names (the encoded names from the fitted
                preprocessor are used when available)

        Returns:
            pd.Series: feature importance values
        """
        pipeline = self.get_fitted(model_name)
        estimator = pipeline.named_steps['model']

        # After encoding, the real feature names come from the preprocessor.
        try:
            names = pipeline.named_steps['preprocess'].get_feature_names_out()
        except Exception:
            names = feature_names

        if hasattr(estimator, 'feature_importances_'):
            return pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False)
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if getattr(coef, 'ndim', 1) > 1:
                # For multi-class, take the mean absolute coefficient
                importance = np.mean(np.abs(coef), axis=0)
            else:
                importance = np.abs(coef)
            return pd.Series(importance, index=names).sort_values(ascending=False)
        else:
            return None