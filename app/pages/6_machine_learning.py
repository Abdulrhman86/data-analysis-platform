# 6_machine_learning.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from config import Config, Paths
from datetime import datetime

# Import our machine learning utilities
from utils.classification_models import ClassificationProcessor
from utils.regression_models import RegressionProcessor
from utils.model_evaluation import get_classification_report, get_confusion_matrix, get_residuals
from utils.model_visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance, plot_residuals, plot_regression_results,
    plot_learning_curve, plot_model_comparison
)
from utils.model_export import create_streamlit_download_button, export_full_pipeline
from utils.feature_selection import select_k_best_features, select_features_with_mutual_info

# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📊",
    page_title="Machine Learning Models"
)

# Apply CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)
st.markdown(Config.stepper(6), unsafe_allow_html=True)

# Initialize session state for models
if 'classification_processor' not in st.session_state:
    st.session_state.classification_processor = ClassificationProcessor()
if 'regression_processor' not in st.session_state:
    st.session_state.regression_processor = RegressionProcessor()
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'current_task' not in st.session_state:
    st.session_state.current_task = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'train_test_data' not in st.session_state:
    st.session_state.train_test_data = {}

# Page header with back navigation
st.markdown('<div class="page-header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
    if st.button('← ', key='back_button', help="Return to previous page"):
        st.switch_page("pages/5_dashboard.py")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="margin: 0; color: white;">Machine Learning Models</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Check if data exists
if 'data' not in st.session_state or st.session_state.data is None:
    st.markdown(Config.empty_state(
        "No data loaded yet",
        "Upload a CSV or Excel file to begin — or load the bundled sample dataset from the Upload page."),
        unsafe_allow_html=True)
    if st.button('Go to Data Upload', use_container_width=True):
        st.switch_page("pages/1_upload_data.py")
    st.stop()

# Get the data
data = st.session_state.data

# Create main tabs
data_prep_tab, model_tab, evaluation_tab, predict_tab, export_tab = st.tabs([
    "Data Preparation", "Model Training", "Model Evaluation", "Predict", "Export Model"
])

with data_prep_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prepare Data for ML</div>', unsafe_allow_html=True)

    # Instructions
    with st.expander("ℹ️ How to prepare your data", expanded=False):
        st.markdown("""
        ## Data Preparation Guide

        Before training machine learning models, you need to prepare your data:

        1. **Select target variable**: Choose the column you want to predict
        2. **Select features**: Choose columns to use for prediction (or use all)
        3. **Choose task type**: Classification (for categorical targets) or Regression (for numeric targets)
        4. **Set test size**: Choose what percentage of data to hold back for testing

        **Tips:**
        - For classification, your target should be categorical (text labels or integers representing classes)
        - For regression, your target should be continuous numeric values
        - Missing values and categorical features should be handled in the preprocessing step
        - The feature selection step can help you identify the most important features
        """)

    # Target selection
    target_column = st.selectbox(
        "Select target column (what you want to predict):",
        data.columns.tolist(),
        index=len(data.columns) - 1  # Default to last column
    )

    # Get target data type
    target_dtype = data[target_column].dtype
    is_numeric_target = pd.api.types.is_numeric_dtype(target_dtype)
    unique_count = data[target_column].nunique()

    # Task selection with automatic suggestion
    task_suggestion = "regression" if is_numeric_target and unique_count > 10 else "classification"
    task_help = (
        "💡 Suggestion: This looks like a regression task (predicting numeric values)"
        if task_suggestion == "regression"
        else "💡 Suggestion: This looks like a classification task (predicting categories)"
    )

    st.info(task_help)

    task_type = st.radio(
        "Select task type:",
        ["Classification", "Regression"],
        index=0 if task_suggestion == "classification" else 1,
        horizontal=True
    )

    # Store current task
    st.session_state.current_task = task_type.lower()
    # Feature selection
    feature_selection_method = st.radio(
        "Feature selection method:",
        ["Use all features", "Select features manually", "Automatic feature selection"],
        horizontal=True
    )

    # Get available features (columns excluding the target)
    available_features = [col for col in data.columns if col != target_column]

    # For regression, we'll identify numerical features separately
    numerical_features = [col for col in available_features
                          if pd.api.types.is_numeric_dtype(data[col])
                          and not pd.api.types.is_bool_dtype(data[col])]
    if feature_selection_method == "Use all features":
        # For regression, only use numerical features
        if st.session_state.current_task == "regression":
            if len(numerical_features) == 0:
                st.error(
                    "No numerical features available for regression. Please convert some features to numeric first.")
                selected_features = []
            else:
                selected_features = numerical_features
                if len(numerical_features) < len(available_features):
                    st.warning(
                        f"Only using {len(numerical_features)} numerical features for regression (excluding categorical features).")
        else:
            # For classification, we can use all features
            selected_features = available_features

    elif feature_selection_method == "Select features manually":
        # For regression, only show numerical features
        if st.session_state.current_task == "regression":
            if len(numerical_features) == 0:
                st.error(
                    "No numerical features available for regression. Please convert some features to numeric first.")
                selectable_features = []
                selected_features = []
            else:
                st.info("For regression tasks, only numerical features can be selected.")
                selectable_features = numerical_features
        else:
            # For classification, show all features
            selectable_features = available_features

        # Show multiselect only if there are features to select
        if len(selectable_features) > 0:
            selected_features = st.multiselect(
                "Select features to use for training:",
                selectable_features,
                default=selectable_features[:min(10, len(selectable_features))]  # Default to 10 features or fewer
            )
        else:
            selected_features = []

    elif feature_selection_method == "Automatic feature selection":
        # First define feature_selector based on task type
        if st.session_state.current_task == "classification":
            feature_selector = st.radio(
                "Feature selection algorithm:",
                ["Mutual Information", "ANOVA F-value"],
                horizontal=True
            )
        else:
            # For regression
            feature_selector = st.radio(
                "Feature selection algorithm:",
                ["Mutual Information", "F-test"],
                horizontal=True
            )

            # Check if we have numerical features for regression
            if len(numerical_features) == 0:
                st.error(
                    "No numerical features available for regression. Please convert some features to numeric first.")

        # Set default number of features based on task
        default_num_features = min(10,
                                   len(numerical_features if st.session_state.current_task == "regression" else available_features))
        max_features = len(numerical_features if st.session_state.current_task == "regression" else available_features)

        num_features = st.slider(
            "Number of features to select:",
            min_value=2 if max_features >= 2 else 1,
            max_value=max_features,
            value=default_num_features
        )

        # Initialize with appropriate default features based on task
        if st.session_state.current_task == "regression":
            selected_features = numerical_features[:num_features]  # Initial default for regression
        else:
            selected_features = available_features[:num_features]  # Initial default for classification

        # Create button for feature selection
        run_selection = st.button("Run Feature Selection")

        if run_selection:
            with st.spinner("Selecting the most important features..."):
                # For regression, only use numerical features
                if st.session_state.current_task == "regression":
                    X = data[numerical_features]
                    feature_pool = numerical_features
                    st.info(f"Using only {len(numerical_features)} numerical features for regression.")
                else:
                    X = data[available_features]
                    feature_pool = available_features

                y = data[target_column]

                # Check for missing values in target column
                missing_target = y.isna().sum()
                if missing_target > 0:
                    st.warning(
                        f"Target column has {missing_target} missing values. These rows will be excluded from feature selection.")

                try:
                    if feature_selector == "Mutual Information":
                        # Run mutual information selection
                        selected_features, scores = select_features_with_mutual_info(
                            X, y, k=num_features, task=st.session_state.current_task
                        )

                        # IMPORTANT: Strictly limit to num_features
                        if len(selected_features) > num_features:
                            selected_features = selected_features[:num_features]

                        # Show feature importance
                        if len(selected_features) > 0:
                            st.success(f"Successfully selected {len(selected_features)} features")

                            # Show feature importance scores
                            st.subheader("Feature Importance Scores")
                            top_scores = scores.sort_values(ascending=False).head(len(selected_features))

                            fig = go.Figure([
                                go.Bar(
                                    y=top_scores.index,
                                    x=top_scores.values,
                                    orientation='h'
                                )
                            ])
                            fig.update_layout(
                                xaxis_title="Mutual Information Score",
                                yaxis_title="Feature",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(
                                "No features could be selected. This may be due to missing values or non-numeric data.")
                    else:
                        # Run F-test or ANOVA selection
                        selected_features = select_k_best_features(
                            X, y, k=num_features, task=st.session_state.current_task
                        )

                        # IMPORTANT: Strictly limit to num_features
                        if len(selected_features) > num_features:
                            selected_features = selected_features[:num_features]

                        if len(selected_features) > 0:
                            st.success(f"Successfully selected {len(selected_features)} features")
                        else:
                            st.error(
                                "No features could be selected. This may be due to missing values or non-numeric data.")

                    # Always display the selected features
                    if len(selected_features) > 0:
                        st.subheader("Selected features:")
                        st.table(pd.DataFrame({'Feature Name': selected_features}))

                except Exception as e:
                    st.error(f"Error during feature selection: {str(e)}")

                    # Fall back to using appropriate features based on task
                    if st.session_state.current_task == "regression":
                        selected_features = numerical_features[:num_features]
                        st.info(f"Falling back to first {len(selected_features)} numerical features.")
                    else:
                        selected_features = available_features[:num_features]
                        st.info(f"Falling back to first {len(selected_features)} features.")

    # Ensure we have feature names stored before proceeding
    if not selected_features:
        st.error("No features selected. Please select at least one feature before proceeding.")

    # Store feature names in session state
    st.session_state.feature_names = selected_features

    # Test size selection
    test_size = st.slider("Test set size (%):", 10, 40, 20) / 100

    # Random state for reproducibility
    random_state = st.number_input("Random seed (for reproducibility):", value=42)

    # Missing values handling section - Added this section
    with st.expander("Missing Values Handling", expanded=True):
        # Check for missing values in both features and target
        target_missing = data[target_column].isna().sum()

        # Get selected feature columns to check
        if st.session_state.current_task == "regression":
            feature_cols_to_check = [col for col in selected_features if col in numerical_features]
        else:
            feature_cols_to_check = selected_features

        # Check missing values in features
        feature_missing = {}
        total_missing = 0
        for col in feature_cols_to_check:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                feature_missing[col] = missing_count
                total_missing += missing_count

        # Display missing value information
        if target_missing > 0 or feature_missing:
            st.warning("Missing values detected in your data")

            if target_missing > 0:
                st.write(
                    f"Target column '{target_column}': {target_missing} missing values ({(target_missing / len(data)) * 100:.2f}%)")

            if feature_missing:
                st.write("Features with missing values:")
                missing_df = pd.DataFrame({
                    'Column': list(feature_missing.keys()),
                    'Missing Count': list(feature_missing.values()),
                    'Missing Percentage': [f"{(count / len(data)) * 100:.2f}%" for count in feature_missing.values()]
                })
                st.dataframe(missing_df, use_container_width=True)

            # Handling options
            missing_handling = st.radio(
                "How to handle missing values:",
                ["Remove rows with missing values", "Fill missing values automatically"],
                index=0 if (total_missing / (len(data) * len(feature_cols_to_check))) < 0.05 else 1
            )

            if missing_handling == "Remove rows with missing values":
                st.info("Rows with missing values will be removed during data preparation.")
                st.session_state.missing_handling = "remove"
            else:
                if st.session_state.current_task == "regression":
                    st.info("Missing values in features will be filled with column means.")
                else:
                    st.info("Missing values in features will be filled with column modes (most frequent values).")
                st.session_state.missing_handling = "impute"

            st.write("ℹ️ Note: For more advanced handling of missing values, please use the preprocessing page.")
        else:
            st.success("No missing values detected in the selected columns.")
            st.session_state.missing_handling = None

    # Prepare data button
    if st.button("Prepare Data"):
        with st.spinner("Preparing training and test data..."):
            try:
                if st.session_state.current_task == "classification":
                    processor = st.session_state.classification_processor
                else:
                    processor = st.session_state.regression_processor

                # Create a copy of the data for preparation
                prep_data = data.copy()

                # Handle missing values if present
                if hasattr(st.session_state, 'missing_handling') and st.session_state.missing_handling:
                    if st.session_state.missing_handling == "remove":
                        # Remove rows with missing values in either target or selected features
                        missing_mask = prep_data[target_column].isna() | prep_data[selected_features].isna().any(axis=1)
                        rows_before = len(prep_data)
                        prep_data = prep_data[~missing_mask]
                        rows_removed = rows_before - len(prep_data)

                        if rows_removed > 0:
                            st.info(f"Removed {rows_removed} rows with missing values.")

                    elif st.session_state.missing_handling == "impute":
                        # Feature missing values are imputed INSIDE the training
                        # pipeline (median for numeric, most-frequent for
                        # categorical), fit on the training split only -- so we do
                        # not impute features here, which would leak test info.
                        st.info(
                            "Missing feature values will be imputed during training "
                            "(fit on the training split only -- no data leakage)."
                        )

                        # The target cannot be imputed by the feature pipeline, so
                        # handle it here.
                        if prep_data[target_column].isna().any():
                            if st.session_state.current_task == "regression":
                                fill_value = prep_data[target_column].mean()
                                prep_data[target_column] = prep_data[target_column].fillna(fill_value)
                                st.info(f"Filled missing values in target column with mean: {fill_value:.4f}")
                            else:
                                fill_value = prep_data[target_column].mode().iloc[0] if not prep_data[
                                    target_column].mode().empty else "UNKNOWN"
                                prep_data[target_column] = prep_data[target_column].fillna(fill_value)
                                st.info(f"Filled missing values in target column with mode: {fill_value}")

                # Prepare data
                train_test_data = processor.prepare_data(
                    prep_data,
                    target_column=target_column,
                    feature_columns=selected_features,
                    test_size=test_size,
                    random_state=random_state
                )
                # Store data in session state
                st.session_state.feature_names = train_test_data['X_train'].columns.tolist()
                st.session_state.train_test_data = train_test_data

                # Show data summary
                st.success("Data preparation complete! Ready for model training.")
                st.write(f"Training set: {len(train_test_data['X_train'])} samples")
                st.write(f"Testing set: {len(train_test_data['X_test'])} samples")

                # Important message about features
                actual_features = train_test_data['feature_names']
                original_count = len(selected_features)
                final_count = len(actual_features)

                if final_count < original_count:
                    st.warning(
                        f"Only {final_count} out of {original_count} selected features were used (non-numeric features were removed)")

                st.write(f"Number of features: {len(actual_features)}")

                # Show sample of training data
                with st.expander("Preview Training Data", expanded=False):
                    preview_df = pd.concat([train_test_data['X_train'].reset_index(drop=True),
                                            train_test_data['y_train'].reset_index(drop=True)], axis=1)
                    st.dataframe(preview_df.head(10), use_container_width=True)

            except ValueError as e:
                st.error(f"Error preparing data: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

with model_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Train Machine Learning Models</div>', unsafe_allow_html=True)

    # Instructions
    with st.expander("ℹ️ How to train models", expanded=False):
        st.markdown("""
        ## Model Training Guide

        After preparing your data, you can train machine learning models:

        1. **Select model**: Choose from the available models for your task type
        2. **Configure parameters**: Adjust the hyperparameters to optimize model performance
        3. **Train model**: Click the "Train Model" button to fit the model to your data
        4. **Compare models**: Train multiple models to compare their performance

        **Tips:**
        - Start with simple models before trying complex ones
        - Different models have different strengths and weaknesses
        - No single model is best for all datasets
        - Training time increases with dataset size and model complexity
        """)

    # Check if data preparation was completed
    if not st.session_state.train_test_data:
        st.warning("Please complete the data preparation step first.")
    else:
        # Get available models based on task type
        if st.session_state.current_task == "classification":
            processor = st.session_state.classification_processor
            model_options = list(processor.models.keys())
        else:
            processor = st.session_state.regression_processor
            model_options = list(processor.models.keys())

        # Model selection
        selected_model = st.selectbox("Select model to train:", model_options)

        # Show model description
        model_descriptions = {
            "Logistic Regression": """
                **Logistic Regression** is a linear model for classification that estimates the probability of a binary outcome.

                **Strengths:**
                - Simple and interpretable
                - Works well with linearly separable data
                - Less prone to overfitting in high-dimensional spaces

                **Weaknesses:**
                - May underperform with non-linear relationships
                - Limited expressiveness for complex problems

                **Best for:**
                - Binary classification problems
                - Understanding feature influence
                - When explainability is important
            """,
            "Decision Tree": """
                **Decision Tree** is a non-parametric model that learns decision rules from data features.

                **Strengths:**
                - Handles both numerical and categorical data
                - Captures non-linear relationships
                - Easy to interpret and visualize

                **Weaknesses:**
                - Prone to overfitting, especially with deep trees
                - Can be unstable (small data changes may result in very different trees)

                **Best for:**
                - When you need a visual representation of decision rules
                - When feature interactions are important
                - As a building block for more advanced models
            """,
            "Random Forest": """
                **Random Forest** combines multiple decision trees to improve prediction accuracy and control overfitting.

                **Strengths:**
                - Usually performs very well out of the box
                - Handles high-dimensional data well
                - Provides feature importance measures
                - Resistant to overfitting

                **Weaknesses:**
                - Less interpretable than single decision trees
                - Computationally more intensive
                - May be overkill for simple, linear problems

                **Best for:**
                - Complex classification problems
                - When you have many features
                - When accuracy is more important than interpretation
            """,
            "SVM": """
                **Support Vector Machine (SVM)** finds the hyperplane that best separates classes with maximum margin.

                **Strengths:**
                - Effective in high-dimensional spaces
                - Memory efficient
                - Versatile through different kernel functions

                **Weaknesses:**
                - Does not scale well to larger datasets
                - Requires careful tuning of parameters
                - Less interpretable

                **Best for:**
                - Text classification
                - Image classification
                - When the boundary between classes is clear
            """,
            "Naive Bayes": """
                **Naive Bayes** applies Bayes' theorem with strong independence assumptions between features.

                **Strengths:**
                - Fast training and prediction
                - Works well with high-dimensional data
                - Good with small training sets

                **Weaknesses:**
                - "Naive" assumption of feature independence
                - Can be outperformed by more sophisticated models

                **Best for:**
                - Text classification (spam filtering, document categorization)
                - Quick baseline model
                - When features are relatively independent
            """,
            "Linear Regression": """
                **Linear Regression** models the relationship between a dependent variable and one or more independent variables using a linear equation.

                **Strengths:**
                - Simple and interpretable
                - Fast to train and make predictions
                - Weights directly show feature importance

                **Weaknesses:**
                - Only captures linear relationships
                - Sensitive to outliers
                - Assumes independence of features

                **Best for:**
                - Simple predictive tasks
                - When relationship is approximately linear
                - When interpretability is important
            """,
            "Ridge Regression": """
                **Ridge Regression** adds L2 regularization to linear regression to prevent overfitting.

                **Strengths:**
                - Handles multicollinearity well
                - Prevents overfitting
                - Keeps all features but reduces their impact

                **Weaknesses:**
                - Still assumes linear relationship
                - Requires tuning of regularization parameter

                **Best for:**
                - When features are correlated
                - When all features should be kept in the model
                - Improving upon simple linear regression
            """,
            "Lasso Regression": """
                **Lasso Regression** adds L1 regularization to linear regression, which can lead to feature selection.

                **Strengths:**
                - Can produce sparse models (feature selection)
                - Prevents overfitting
                - Reduces model complexity

                **Weaknesses:**
                - Still assumes linear relationship
                - May be unstable with highly correlated features

                **Best for:**
                - Feature selection
                - When a simpler model is preferred
                - When some features are likely irrelevant
            """,
            "Decision Tree Regressor": """
                **Decision Tree Regressor** predicts values based on decision rules inferred from the data features.

                **Strengths:**
                - Captures non-linear relationships
                - No assumptions about data distribution
                - Handles numerical and categorical data

                **Weaknesses:**
                - Prone to overfitting
                - Can be unstable
                - Not always optimal for simple relationships

                **Best for:**
                - Complex non-linear relationships
                - When a visual representation of decisions is helpful
                - Understanding feature interactions
            """,
            "Random Forest Regressor": """
                **Random Forest Regressor** combines multiple decision trees to improve prediction accuracy and control overfitting.

                **Strengths:**
                - Usually performs very well out of the box
                - Resistant to overfitting
                - Handles many features well
                - Provides feature importance

                **Weaknesses:**
                - Less interpretable than single decision trees
                - Computationally intensive
                - Overkill for simple problems

                **Best for:**
                - Complex regression problems
                - When you have many features
                - When accuracy is more important than interpretation
            """
        }

        if selected_model in model_descriptions:
            with st.expander("About this model", expanded=False):
                st.markdown(model_descriptions[selected_model])

        # Get current model configuration
        current_config = processor.model_configs.get(selected_model, {})

        # Show model parameters based on model type
        with st.expander("Model Parameters", expanded=True):
            new_params = {}

            if selected_model == "Logistic Regression":
                col1, col2 = st.columns(2)
                with col1:
                    new_params['C'] = st.number_input(
                        "Regularization strength (C):",
                        min_value=0.01,
                        max_value=10.0,
                        value=float(current_config.get('C', 1.0)),
                        help="Smaller values specify stronger regularization"
                    )
                with col2:
                    new_params['penalty'] = st.selectbox(
                        "Penalty type:",
                        options=['l2', 'l1', 'elasticnet', 'none'],
                        index=['l2', 'l1', 'elasticnet', 'none'].index(current_config.get('penalty', 'l2')),
                        help="l1 is Lasso, l2 is Ridge, elasticnet is combination"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    new_params['solver'] = st.selectbox(
                        "Solver algorithm:",
                        options=['lbfgs', 'liblinear', 'newton-cg', 'saga'],
                        index=['lbfgs', 'liblinear', 'newton-cg', 'saga'].index(current_config.get('solver', 'lbfgs')),
                        help="Algorithm to use in the optimization problem"
                    )
                with col2:
                    new_params['max_iter'] = st.number_input(
                        "Maximum iterations:",
                        min_value=100,
                        max_value=1000,
                        value=int(current_config.get('max_iter', 100)),
                        step=100,
                        help="Maximum number of iterations for solver"
                    )

            elif "Decision Tree" in selected_model:
                col1, col2 = st.columns(2)
                with col1:
                    new_params['max_depth'] = st.number_input(
                        "Maximum depth:",
                        min_value=1,
                        max_value=30,
                        value=int(current_config.get('max_depth', 5)),
                        help="Maximum depth of the tree"
                    )
                with col2:
                    new_params['min_samples_split'] = st.number_input(
                        "Minimum samples to split:",
                        min_value=2,
                        max_value=20,
                        value=int(current_config.get('min_samples_split', 2)),
                        help="Minimum samples required to split an internal node"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    new_params['min_samples_leaf'] = st.number_input(
                        "Minimum samples per leaf:",
                        min_value=1,
                        max_value=20,
                        value=int(current_config.get('min_samples_leaf', 1)),
                        help="Minimum samples required to be at a leaf node"
                    )
                with col2:
                    criterion_options = ['gini', 'entropy'] if "Regressor" not in selected_model else ['squared_error',
                                                                                                       'absolute_error']
                    default_criterion = current_config.get('criterion', criterion_options[0])
                    new_params['criterion'] = st.selectbox(
                        "Split criterion:",
                        options=criterion_options,
                        index=criterion_options.index(
                            default_criterion) if default_criterion in criterion_options else 0,
                        help="Function to measure quality of a split"
                    )

            elif "Random Forest" in selected_model:
                col1, col2 = st.columns(2)
                with col1:
                    new_params['n_estimators'] = st.number_input(
                        "Number of trees:",
                        min_value=10,
                        max_value=500,
                        value=int(current_config.get('n_estimators', 100)),
                        step=10,
                        help="Number of trees in the forest"
                    )
                with col2:
                    new_params['max_depth'] = st.number_input(
                        "Maximum depth:",
                        min_value=1,
                        max_value=30,
                        value=int(current_config.get('max_depth', 5)),
                        help="Maximum depth of each tree"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    new_params['min_samples_split'] = st.number_input(
                        "Minimum samples to split:",
                        min_value=2,
                        max_value=20,
                        value=int(current_config.get('min_samples_split', 2)),
                        help="Minimum samples required to split an internal node"
                    )
                with col2:
                    new_params['min_samples_leaf'] = st.number_input(
                        "Minimum samples per leaf:",
                        min_value=1,
                        max_value=20,
                        value=int(current_config.get('min_samples_leaf', 1)),
                        help="Minimum samples required to be at a leaf node"
                    )

                criterion_options = ['gini', 'entropy'] if "Regressor" not in selected_model else ['squared_error',
                                                                                                   'absolute_error']
                default_criterion = current_config.get('criterion', criterion_options[0])
                new_params['criterion'] = st.selectbox(
                    "Split criterion:",
                    options=criterion_options,
                    index=criterion_options.index(default_criterion) if default_criterion in criterion_options else 0,
                    help="Function to measure quality of a split"
                )

            elif selected_model == "SVM":
                col1, col2 = st.columns(2)
                with col1:
                    new_params['C'] = st.number_input(
                        "Regularization parameter (C):",
                        min_value=0.1,
                        max_value=10.0,
                        value=float(current_config.get('C', 1.0)),
                        help="Penalty parameter of the error term"
                    )
                with col2:
                    new_params['kernel'] = st.selectbox(
                        "Kernel type:",
                        options=['rbf', 'linear', 'poly', 'sigmoid'],
                        index=['rbf', 'linear', 'poly', 'sigmoid'].index(current_config.get('kernel', 'rbf')),
                        help="Kernel function to use"
                    )

                if new_params['kernel'] in ['rbf', 'poly', 'sigmoid']:
                    new_params['gamma'] = st.selectbox(
                        "Gamma (kernel coefficient):",
                        options=['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                        index=['scale', 'auto', 0.001, 0.01, 0.1, 1.0].index(current_config.get('gamma', 'scale')),
                        help="Kernel coefficient"
                    )

            elif selected_model == "Naive Bayes":
                # GaussianNB has minimal parameters to tune
                new_params['var_smoothing'] = st.number_input(
                    "Variance smoothing:",
                    min_value=1e-10,
                    max_value=1e-5,
                    value=float(current_config.get('var_smoothing', 1e-9)),
                    format="%.10f",
                    help="Portion of the largest variance of all features that is added to variances for calculation stability"
                )

            elif "Linear Regression" in selected_model:
                st.info("Linear Regression has no hyperparameters to tune.")
                new_params = {}

            elif "Ridge" in selected_model:
                new_params['alpha'] = st.number_input(
                    "Regularization strength (alpha):",
                    min_value=0.01,
                    max_value=10.0,
                    value=float(current_config.get('alpha', 1.0)),
                    help="Regularization strength; larger values specify stronger regularization"
                )
                new_params['solver'] = st.selectbox(
                    "Solver:",
                    options=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    index=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'].index(
                        current_config.get('solver', 'auto')),
                    help="Solver to use in the computational routines"
                )

            elif "Lasso" in selected_model:
                new_params['alpha'] = st.number_input(
                    "Regularization strength (alpha):",
                    min_value=0.01,
                    max_value=10.0,
                    value=float(current_config.get('alpha', 1.0)),
                    help="Regularization strength; larger values specify stronger regularization"
                )
                new_params['max_iter'] = st.number_input(
                    "Maximum iterations:",
                    min_value=100,
                    max_value=10000,
                    value=int(current_config.get('max_iter', 1000)),
                    step=100,
                    help="Maximum number of iterations"
                )

            # Update model parameters
            if new_params and st.button("Update Model Parameters"):
                processor.update_model_parameters(selected_model, new_params)
                st.success(f"Parameters updated for {selected_model}")

        # Training options
        with st.expander("Training Options", expanded=True):
            scaler_choice = st.selectbox(
                "Numeric feature scaling:",
                options=["standard", "robust", "minmax", "power"],
                help="How numeric features are scaled inside the training pipeline (fit on the training split only)."
            )
            processor.preprocessing['scaler'] = scaler_choice

            tune = st.checkbox(
                "Tune hyperparameters (grid search)", value=False,
                help="Search a small grid of hyperparameters with cross-validation and keep the best model."
            )

            use_cv = st.checkbox("Use cross-validation", value=True,
                                 help="Cross-validation trains the model on different subsets of the data to ensure robust performance")

            if use_cv:
                cv_folds = st.slider("Number of cross-validation folds:", 3, 10, 5,
                                     help="Higher numbers give more robust estimates but take longer to train")

        # Train model button
        if st.button(f"Train {selected_model}"):
            with st.spinner(f"Training {selected_model}..."):
                train_test_data = st.session_state.train_test_data
                # Extract training data
                X_train = train_test_data['X_train']
                y_train = train_test_data['y_train']
                X_test = train_test_data['X_test']
                y_test = train_test_data['y_test']

                # Train model (optionally tuning hyperparameters first)
                if tune:
                    tune_scoring = ('accuracy' if st.session_state.current_task == 'classification'
                                    else 'neg_mean_squared_error')
                    model, best_params = processor.tune_model(
                        selected_model, X_train, y_train,
                        cv=cv_folds if use_cv else 5, scoring=tune_scoring
                    )
                    if best_params:
                        st.info(f"Best hyperparameters: {best_params}")
                else:
                    model = processor.train_model(selected_model, X_train, y_train)

                # Store trained model
                st.session_state.trained_models[selected_model] = model

                # Cross-validation if requested
                if use_cv:
                    with st.spinner(f"Performing {cv_folds}-fold cross-validation..."):
                        if st.session_state.current_task == "classification":
                            scoring = 'accuracy'
                        else:
                            scoring = 'neg_mean_squared_error'

                        cv_scores = processor.cross_validate(
                            selected_model,
                            X_train,
                            y_train,
                            cv=cv_folds,
                            scoring=scoring
                        )

                        if scoring == 'neg_mean_squared_error':
                            cv_scores = -cv_scores  # Convert to positive MSE
                            cv_score_name = "MSE"
                        else:
                            cv_score_name = scoring.capitalize()

                        st.write(f"Cross-validation {cv_score_name}:")
                        st.write(f"Mean: {cv_scores.mean():.4f}")
                        st.write(f"Standard Deviation: {cv_scores.std():.4f}")

                # Evaluate model
                metrics = processor.evaluate_model(selected_model, X_test, y_test)

                # Store results for comparison
                st.session_state.model_results[selected_model] = metrics

                # Show basic results
                st.success(f"Model trained successfully!")

                if st.session_state.current_task == "classification":
                    st.write(f"Test Accuracy: {metrics['accuracy']:.4f}")
                    st.write(f"F1 Score: {metrics['f1']:.4f}")
                else:
                    st.write(f"Test RMSE: {metrics['rmse']:.4f}")
                    st.write(f"R² Score: {metrics['r2']:.4f}")

                st.write("Go to the 'Model Evaluation' tab for detailed performance metrics.")

    st.markdown('</div>', unsafe_allow_html=True)

with evaluation_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evaluate Model Performance</div>', unsafe_allow_html=True)

    # Instructions
    with st.expander("ℹ️ How to evaluate models", expanded=False):
        st.markdown("""
        ## Model Evaluation Guide

        After training your models, you need to evaluate their performance:

        1. **Select model**: Choose a trained model to evaluate
        2. **View metrics**: Examine the performance metrics specific to your task
        3. **Visualize results**: Study the visualizations to better understand model behavior
        4. **Compare models**: Compare different models to select the best one

        **Metrics for Classification:**
        - **Accuracy**: Proportion of correct predictions (higher is better)
        - **Precision**: Ability to avoid false positives (higher is better)
        - **Recall**: Ability to find all positive instances (higher is better)
        - **F1 Score**: Harmonic mean of precision and recall (higher is better)
        - **ROC AUC**: Area under the ROC curve (higher is better)

        **Metrics for Regression:**
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **R²**: Coefficient of determination (higher is better, max 1.0)

        **The best model depends on your specific goals and requirements.**
        """)

    # Check if any models have been trained
    if not st.session_state.trained_models:
        st.warning("No models trained yet. Please train a model first.")
    else:
        # Model selection
        trained_model_names = list(st.session_state.trained_models.keys())
        model_to_evaluate = st.selectbox("Select model to evaluate:", trained_model_names)

        if model_to_evaluate:
            model = st.session_state.trained_models[model_to_evaluate]
            metrics = st.session_state.model_results.get(model_to_evaluate, {})

            if not metrics:
                st.warning(f"No evaluation metrics found for {model_to_evaluate}. Please retrain the model.")
            else:
                # Extract test data
                train_test_data = st.session_state.train_test_data
                X_test = train_test_data['X_test']
                y_test = train_test_data['y_test']

                # Create tabs for different types of evaluation
                metrics_tab, visual_tab, feature_tab, compare_tab = st.tabs([
                    "Performance Metrics", "Visualizations", "Feature Importance", "Model Comparison"
                ])

                with metrics_tab:
                    # Display metrics based on task type
                    if st.session_state.current_task == "classification":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if "accuracy" in metrics:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        with col2:
                            if "precision" in metrics:
                                st.metric("Precision", f"{metrics['precision']:.4f}")
                        with col3:
                            if "recall" in metrics:
                                st.metric("Recall", f"{metrics['recall']:.4f}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if "f1" in metrics:
                                st.metric("F1 Score", f"{metrics['f1']:.4f}")
                        with col2:
                            if "roc_auc" in metrics:
                                if metrics['roc_auc'] is not None:
                                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")

                        # Classification report
                        st.subheader("Detailed Classification Report")
                        report = get_classification_report(model, X_test, y_test)
                        st.dataframe(report, use_container_width=True)

                    else:  # Regression
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col2:
                            st.metric("MAE", f"{metrics['mae']:.4f}")
                        with col3:
                            st.metric("R² Score", f"{metrics['r2']:.4f}")

                        st.subheader("Error Distribution")
                        residuals = get_residuals(model, X_test, y_test)

                        # Create histogram of residuals
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=residuals,
                            nbinsx=30,
                            marker=dict(color='blue', opacity=0.7)
                        ))
                        fig.update_layout(
                            title="Distribution of Prediction Errors",
                            xaxis_title="Error",
                            yaxis_title="Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with visual_tab:
                    # Visualizations based on task type
                    if st.session_state.current_task == "classification":
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = get_confusion_matrix(model, X_test, y_test)
                        fig = plot_confusion_matrix(cm)
                        st.plotly_chart(fig, use_container_width=True)

                        # Check if binary classification for ROC and PR curves
                        unique_classes = len(np.unique(y_test))
                        if unique_classes == 2:
                            try:
                                # ROC Curve
                                st.subheader("ROC Curve")
                                from utils.model_evaluation import get_roc_curve_data

                                roc_data = get_roc_curve_data(model, X_test, y_test)
                                fig = plot_roc_curve(roc_data)
                                st.plotly_chart(fig, use_container_width=True)

                                # Precision-Recall Curve
                                st.subheader("Precision-Recall Curve")
                                from utils.model_evaluation import get_precision_recall_curve_data

                                pr_data = get_precision_recall_curve_data(model, X_test, y_test)
                                fig = plot_precision_recall_curve(pr_data)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating curves: {str(e)}")
                        else:
                            st.info("ROC and Precision-Recall curves are only available for binary classification.")

                        # Learning Curve
                        st.subheader("Learning Curve")
                        try:
                            from utils.model_evaluation import get_learning_curve_data

                            learning_curve_data = get_learning_curve_data(
                                model,
                                train_test_data['X_train'],
                                train_test_data['y_train']
                            )
                            fig = plot_learning_curve(learning_curve_data)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating learning curve: {str(e)}")

                    else:  # Regression
                        # Residuals Plot
                        st.subheader("Residuals Plot")
                        residuals = get_residuals(model, X_test, y_test)
                        y_pred = model.predict(X_test)
                        fig = plot_residuals(residuals, y_pred)
                        st.plotly_chart(fig, use_container_width=True)

                        # Actual vs Predicted
                        st.subheader("Actual vs Predicted Values")
                        fig = plot_regression_results(y_test, y_pred)
                        st.plotly_chart(fig, use_container_width=True)

                        # Learning Curve
                        st.subheader("Learning Curve")
                        try:
                            from utils.model_evaluation import get_learning_curve_data

                            learning_curve_data = get_learning_curve_data(
                                model,
                                train_test_data['X_train'],
                                train_test_data['y_train']
                            )
                            fig = plot_learning_curve(learning_curve_data, "Learning Curve (MSE)")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating learning curve: {str(e)}")

                with feature_tab:
                    # Feature importance if available
                    st.subheader("Feature Importance")

                    if st.session_state.current_task == "classification":
                        processor = st.session_state.classification_processor
                    else:
                        processor = st.session_state.regression_processor

                    feature_importance = processor.get_feature_importance(
                        model_to_evaluate,
                        st.session_state.feature_names
                    )

                    if feature_importance is not None:
                        # Plot feature importance
                        top_n = st.slider("Number of features to display:", 5, min(20, len(feature_importance)), 10)
                        fig = plot_feature_importance(feature_importance, top_n=top_n)
                        st.plotly_chart(fig, use_container_width=True)

                        # Show feature importance as dataframe
                        st.subheader("Feature Importance Values")
                        st.dataframe(feature_importance.reset_index().rename(
                            columns={"index": "Feature", 0: "Importance"}
                        ).sort_values("Importance", ascending=False), use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type.")

                with compare_tab:
                    # Compare multiple models
                    if len(st.session_state.model_results) > 1:
                        st.subheader("Model Comparison")

                        if st.session_state.current_task == "classification":
                            metrics_to_compare = ["accuracy", "precision", "recall", "f1"]
                            if all('roc_auc' in results and results['roc_auc'] is not None
                                   for results in st.session_state.model_results.values()):
                                metrics_to_compare.append("roc_auc")

                            selected_metric = st.selectbox(
                                "Select metric for comparison:",
                                metrics_to_compare,
                                index=0
                            )
                            higher_is_better = True
                        else:
                            metrics_to_compare = ["rmse", "mae", "r2"]
                            selected_metric = st.selectbox(
                                "Select metric for comparison:",
                                metrics_to_compare,
                                index=0
                            )
                            higher_is_better = selected_metric == "r2"  # Only R² is better when higher

                        fig = plot_model_comparison(
                            st.session_state.model_results,
                            selected_metric,
                            higher_is_better
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show metrics table
                        comparison_data = []
                        for model_name, results in st.session_state.model_results.items():
                            model_metrics = {
                                "Model": model_name
                            }
                            for metric in metrics_to_compare:
                                if metric in results:
                                    model_metrics[metric] = results[metric]
                            comparison_data.append(model_metrics)

                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)

                        # Highlight best model
                        best_model_idx = comparison_df[selected_metric].argmax() if higher_is_better else comparison_df[
                            selected_metric].argmin()
                        best_model = comparison_df.iloc[best_model_idx]["Model"]
                        best_value = comparison_df.iloc[best_model_idx][selected_metric]

                        st.success(f"Best model based on {selected_metric}: {best_model} ({best_value:.4f})")
                    else:
                        st.info("Train multiple models to compare their performance.")

    st.markdown('</div>', unsafe_allow_html=True)

with predict_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predict on New Data</div>', unsafe_allow_html=True)

    if not st.session_state.trained_models:
        st.warning("No models trained yet. Train a model first.")
    else:
        predict_model_name = st.selectbox(
            "Model to use:", list(st.session_state.trained_models.keys()), key="predict_model_select"
        )
        st.caption("Upload a file with the same feature columns used for training. "
                   "Missing values and categorical encoding are handled by the model's pipeline.")
        new_file = st.file_uploader(
            "Upload new data (CSV or Excel):", type=["csv", "xlsx", "xls"], key="predict_upload"
        )

        if new_file is not None and predict_model_name:
            try:
                if new_file.name.lower().endswith('.csv'):
                    new_df = pd.read_csv(new_file)
                else:
                    new_df = pd.read_excel(new_file)
                new_df.columns = new_df.columns.astype(str)

                model = st.session_state.trained_models[predict_model_name]
                features = st.session_state.get('feature_names', [])
                missing = [c for c in features if c not in new_df.columns]
                if missing:
                    st.error(f"Uploaded data is missing required feature columns: {missing}")
                else:
                    X_new = new_df[features] if features else new_df
                    predictions = model.predict(X_new)
                    result = new_df.copy()
                    result['prediction'] = predictions
                    st.success(f"Generated {len(predictions)} predictions.")
                    st.dataframe(result.head(20), use_container_width=True)
                    st.download_button(
                        "Download predictions (CSV)",
                        data=result.to_csv(index=False).encode('utf-8'),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Could not generate predictions: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

with export_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Export Trained Model</div>', unsafe_allow_html=True)

    # Instructions
    with st.expander("ℹ️ How to export models", expanded=False):
        st.markdown("""
        ## Model Export Guide

        After training and evaluating models, you can export the best one:

        1. **Select model**: Choose the model you want to export
        2. **Add metadata**: Include information about the model and its performance
        3. **Download model**: Save the model file for future use

        **Uses for exported models:**
        - Deploy in production environments
        - Make predictions on new data
        - Share with colleagues
        - Archive for documentation

        **The exported model package includes:**
        - The trained model file (.pkl)
        - Metadata about features and performance
        """)

    # Check if any models have been trained
    if not st.session_state.trained_models:
        st.warning("No models trained yet. Please train a model first.")
    else:
        # Model selection
        trained_model_names = list(st.session_state.trained_models.keys())
        model_to_export = st.selectbox("Select model to export:", trained_model_names, key="export_model_select")

        if model_to_export:
            # Get model and its metrics
            model = st.session_state.trained_models[model_to_export]
            metrics = st.session_state.model_results.get(model_to_export, {})

            # Additional metadata
            st.subheader("Model Metadata")
            model_version = st.text_input("Model Version:", "1.0.0")
            model_description = st.text_area("Model Description:",
                                             f"{model_to_export} trained on {st.session_state.train_test_data['X_train'].shape[0]} samples")

            # Prepare metadata
            metadata = {
                "model_name": model_to_export,
                "model_version": model_version,
                "model_description": model_description,
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task_type": st.session_state.current_task,
                "features": st.session_state.feature_names,
                "metrics": metrics,
                "data_shape": {
                    "train_samples": st.session_state.train_test_data['X_train'].shape[0],
                    "test_samples": st.session_state.train_test_data['X_test'].shape[0],
                    "features": len(st.session_state.feature_names)
                }
            }

            # Export options
            export_type = st.radio("Export type:", ["Single Model", "Full Pipeline"], horizontal=True)

            if export_type == "Single Model":
                # Download button for the model
                st.subheader("Download Model")
                create_streamlit_download_button(model, model_to_export, metadata)
            else:
                # Export full pipeline
                st.subheader("Download Full Pipeline")
                if 'pipelines' in st.session_state and st.session_state.current_pipeline:
                    # Include preprocessing steps if available
                    pipeline = st.session_state.pipelines[st.session_state.current_pipeline]
                    preprocessing_steps = [step.to_dict() for step in pipeline.steps]
                    metadata["preprocessing_pipeline"] = {
                        "pipeline_name": pipeline.name,
                        "steps_count": len(preprocessing_steps)
                    }
                else:
                    preprocessing_steps = None

                # Create a dict of models if multiple are trained
                models_to_export = {}
                include_all_models = st.checkbox("Include all trained models", value=False)

                if include_all_models:
                    models_to_export = st.session_state.trained_models
                else:
                    models_to_export = {model_to_export: model}

                pipeline_buffer = export_full_pipeline(models_to_export, metadata, preprocessing_steps)

                st.download_button(
                    label="Download Complete ML Pipeline",
                    data=pipeline_buffer,
                    file_name="ml_pipeline.zip",
                    mime="application/zip"
                )

            st.success("Model ready for export!")

    st.markdown('</div>', unsafe_allow_html=True)

# Navigation at the bottom
st.divider()
col1, col2 = st.columns([1, 1])
with col1:
    if st.button('← Previous: Dashboards', key='prev_button'):
        st.switch_page("pages/5_dashboard.py")
with col2:
    pass  # Future page could go here