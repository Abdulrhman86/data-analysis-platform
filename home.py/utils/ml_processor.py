import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import pickle


class MLProcessor:
    """Base class for machine learning operations.

    Every model is trained inside an sklearn ``Pipeline`` whose first step is a
    ``ColumnTransformer`` that imputes + scales the numeric columns and imputes +
    one-hot encodes the categorical columns. The whole pipeline is fit **only on
    the training split**, which:

    * lets categorical features be used without crashing scikit-learn,
    * scales features so distance/gradient models (SVM, LogReg, Ridge, Lasso) work,
    * prevents train/test leakage (scaler/encoder/imputer never see the test set),
    * makes the exported model usable on raw, unprocessed data.
    """

    # Overridden by subclasses ('classification' / 'regression').
    task_type = None

    def __init__(self):
        self.models = {}          # name -> unfitted estimator (as registered)
        self.model_configs = {}   # name -> hyperparameter dict
        self.fitted_models = {}   # name -> fitted sklearn Pipeline
        self.results = {}
        self.preprocessing = {'scaler': 'standard'}  # numeric scaler choice

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_preprocessor(X, scaler='standard'):
        """Build a ColumnTransformer for the columns present in ``X``.

        ``scaler`` selects the numeric scaler: standard / robust / minmax / power.
        """
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()

        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer(),
        }
        numeric_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', scalers.get(scaler, StandardScaler())),
        ])
        categorical_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])

        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_pipe, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_pipe, categorical_features))

        # remainder='drop' ignores anything that is neither numeric nor
        # categorical (e.g. datetime columns should be engineered first).
        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _build_pipeline(self, model_name, X):
        """Wrap a registered estimator in a fresh preprocessing pipeline."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        return Pipeline([
            ('preprocess', self.build_preprocessor(X, scaler=self.preprocessing.get('scaler', 'standard'))),
            ('model', self.models[model_name]),
        ])

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def prepare_data(self, df, target_column, feature_columns=None, test_size=0.2, random_state=42):
        """Split data into train/test sets.

        Returns a dict with X_train, X_test, y_train, y_test, and feature_names.
        Rows with a missing *target* are dropped; missing *feature* values are
        left untouched here and imputed inside the training pipeline (fit on the
        train split only) so that no test information leaks.
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        usable_features = [col for col in feature_columns if col in df.columns]
        if not usable_features:
            raise ValueError("No valid features available for model training.")

        X = df[usable_features].copy()
        y = df[target_column].copy()

        valid_rows = ~y.isna()
        if valid_rows.sum() == 0:
            raise ValueError("No valid rows after removing rows with a missing target.")
        X = X[valid_rows]
        y = y[valid_rows]

        if self.task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target column must be numeric for regression tasks.")

        # Stratify classification splits when every class has at least 2 rows.
        stratify = None
        if self.task_type == 'classification' and y.value_counts().min() >= 2:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': usable_features,
        }

    # ------------------------------------------------------------------ #
    # Model registry / training
    # ------------------------------------------------------------------ #
    def register_model(self, model_name, model_instance, model_config=None):
        """Register an (unfitted) estimator for use."""
        self.models[model_name] = model_instance
        self.model_configs[model_name] = model_config or {}

    def train_model(self, model_name, X_train, y_train):
        """Fit a preprocessing+model pipeline on the training split.

        Returns the fitted ``Pipeline`` (also stored in ``self.fitted_models``).
        """
        pipeline = self._build_pipeline(model_name, X_train)
        pipeline.fit(X_train, y_train)
        self.fitted_models[model_name] = pipeline
        return pipeline

    def get_fitted(self, model_name):
        """Return the fitted pipeline for a model, or raise if not trained."""
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet.")
        return self.fitted_models[model_name]

    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a trained model (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement evaluate_model")

    def cross_validate(self, model_name, X, y, cv=5, scoring=None):
        """Cross-validate a model.

        Builds a fresh pipeline so the preprocessing is re-fit inside every fold
        (no leakage). scikit-learn automatically uses StratifiedKFold for
        classifier pipelines when ``cv`` is an integer.
        """
        estimator = self._build_pipeline(model_name, X)
        return cross_val_score(estimator, X, y, cv=cv, scoring=scoring)

    def get_param_grid(self, model_name):
        """Hyperparameter grid for tuning (overridden by subclasses)."""
        return {}

    def tune_model(self, model_name, X, y, search='grid', cv=5, n_iter=10, scoring=None):
        """Search hyperparameters for a model and keep the best fitted pipeline.

        Returns ``(best_pipeline, best_params)``. Falls back to a plain fit when
        the model has no tunable grid.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        pipeline = self._build_pipeline(model_name, X)
        grid = {f"model__{k}": v for k, v in self.get_param_grid(model_name).items()}
        if not grid:
            pipeline.fit(X, y)
            self.fitted_models[model_name] = pipeline
            return pipeline, {}
        if search == 'random':
            searcher = RandomizedSearchCV(pipeline, grid, n_iter=n_iter, cv=cv,
                                          scoring=scoring, random_state=42)
        else:
            searcher = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring)
        searcher.fit(X, y)
        self.fitted_models[model_name] = searcher.best_estimator_
        best_params = {k.replace('model__', ''): v for k, v in searcher.best_params_.items()}
        return searcher.best_estimator_, best_params

    # ------------------------------------------------------------------ #
    # Persistence (saves the FITTED pipeline, incl. preprocessing)
    # ------------------------------------------------------------------ #
    def save_model(self, model_name, filepath):
        """Save a trained pipeline to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_fitted(model_name), f)

    def load_model(self, model_name, filepath):
        """Load a fitted pipeline from disk and register it as trained.

        SECURITY: ``pickle.load`` executes arbitrary code while unpickling. Only
        load model files you created or fully trust -- never an untrusted or
        user-uploaded file.
        """
        with open(filepath, 'rb') as f:
            self.fitted_models[model_name] = pickle.load(f)
