import pandas as pd
import numpy as np
import streamlit as st
from utils.data_processor import DataProcessor
from utils.preprocessing_pipeline import record_step


class FeatureEngineeringProcessor(DataProcessor):
    """Class for feature engineering operations"""

    def __init__(self):
        super().__init__()

    @record_step("feature")
    def create_polynomial_features(self, df, column, degree=2):
        """Create polynomial features from a numeric column"""
        result_df = df.copy()

        for i in range(2, degree + 1):
            result_df[f"{column}^{i}"] = np.power(df[column], i)

        return result_df

    @record_step("feature")
    def create_interaction_term(self, df, column1, column2, operation='multiply'):
        """Create interaction between two numeric columns"""
        result_df = df.copy()

        if operation == 'multiply':
            result_df[f"{column1}_x_{column2}"] = df[column1] * df[column2]
        elif operation == 'divide':
            # Handle division by zero
            result_df[f"{column1}_div_{column2}"] = df[column1] / df[column2].replace(0, np.nan)
        elif operation == 'add':
            result_df[f"{column1}_plus_{column2}"] = df[column1] + df[column2]
        elif operation == 'subtract':
            result_df[f"{column1}_minus_{column2}"] = df[column1] - df[column2]

        return result_df

    @record_step("feature")
    def create_binned_feature(self, df, column, method='equal_width', bins=4, labels=None):
        """Create a binned feature from a numeric column"""
        result_df = df.copy()

        if method == 'equal_width':
            result_df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
        elif method == 'equal_frequency':
            result_df[f"{column}_binned"] = pd.qcut(df[column], q=bins, labels=labels, duplicates='drop')

        return result_df

    @record_step("feature")
    def create_aggregation_features(self, df, group_col, agg_col, agg_functions=None):
        """Create aggregation features grouped by a categorical column"""
        if agg_functions is None:
            agg_functions = ['mean', 'median', 'min', 'max', 'std']

        result_df = df.copy()

        for func in agg_functions:
            # Calculate aggregation
            agg_values = df.groupby(group_col)[agg_col].agg(func)

            # Map aggregated values back to original dataframe
            result_df[f"{agg_col}_{func}_by_{group_col}"] = result_df[group_col].map(agg_values)

        return result_df

    @record_step("feature")
    def create_rolling_features(self, df, date_col, value_col, windows=None, functions=None):
        """Create rolling window features from time series data"""
        if windows is None:
            windows = [3, 7, 30]
        if functions is None:
            functions = ['mean', 'std']

        # Sort by date
        df_sorted = df.sort_values(by=date_col).copy()
        result_df = df_sorted.copy()

        for window in windows:
            for func in functions:
                if func == 'mean':
                    result_df[f"{value_col}_rolling_{window}_{func}"] = df_sorted[value_col].rolling(
                        window=window).mean()
                elif func == 'std':
                    result_df[f"{value_col}_rolling_{window}_{func}"] = df_sorted[value_col].rolling(
                        window=window).std()
                elif func == 'min':
                    result_df[f"{value_col}_rolling_{window}_{func}"] = df_sorted[value_col].rolling(
                        window=window).min()
                elif func == 'max':
                    result_df[f"{value_col}_rolling_{window}_{func}"] = df_sorted[value_col].rolling(
                        window=window).max()

        # Reset to original order
        result_df = result_df.loc[df.index]
        return result_df

    @record_step("feature")
    def create_lag_features(self, df, date_col, value_col, lags=None):
        """Create lag features from time series data"""
        if lags is None:
            lags = [1, 7, 30]

        # Sort by date
        df_sorted = df.sort_values(by=date_col).copy()
        result_df = df_sorted.copy()

        for lag in lags:
            result_df[f"{value_col}_lag_{lag}"] = df_sorted[value_col].shift(lag)

        # Reset to original order
        result_df = result_df.loc[df.index]
        return result_df

    @record_step("feature")
    def encode_cyclical_features(self, df, column, max_value=None):
        """Encode cyclical features like hour, day of week, month using sine and cosine"""
        result_df = df.copy()

        if max_value is None:
            if "hour" in column.lower():
                max_value = 24
            elif "day" in column.lower() and "week" in column.lower():
                max_value = 7
            elif "month" in column.lower():
                max_value = 12
            elif "day" in column.lower():
                max_value = 31
            else:
                raise ValueError("Please specify max_value for the cyclical feature")

        # Normalize to [0, 2π]
        angle = 2 * np.pi * df[column] / max_value

        # Create sine and cosine features
        result_df[f"{column}_sin"] = np.sin(angle)
        result_df[f"{column}_cos"] = np.cos(angle)

        return result_df

    @record_step("feature")
    def create_text_features(self, df, text_column):
        """Extract features from text column"""
        result_df = df.copy()

        # Basic text features
        result_df[f"{text_column}_length"] = df[text_column].astype(str).apply(len)
        result_df[f"{text_column}_word_count"] = df[text_column].astype(str).apply(lambda x: len(x.split()))
        result_df[f"{text_column}_capital_ratio"] = df[text_column].astype(str).apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)

        return result_df



def feature_engineering_ui(st, df, column_types):
    """Streamlit UI for feature engineering"""
    st.subheader("Feature Engineering")

    # Initialize feature engineering processor
    fe_processor = FeatureEngineeringProcessor()

    # Separate columns by type
    numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
    categorical_columns = [col for col, type_ in column_types.items() if type_ == 'categorical']
    datetime_columns = [col for col, type_ in column_types.items() if type_ == 'datetime']

    # Feature engineering explanations
    feature_explanations = {
        "Polynomial Features": """
        **Polynomial Features** create new features by raising numeric variables to different powers (squared, cubed, etc.).

        **Example:** If you have a column 'x' with values [1, 2, 3], creating degree 2 polynomial features would add 'x^2' with values [1, 4, 9].

        **Use case:** Helpful when the relationship between variables is non-linear.
        """,

        "Interaction Features": """
        **Interaction Features** create new features by combining two existing numeric variables using operations like multiplication, division, addition, or subtraction.

        **Example:** If you have 'height' and 'width', you might create 'height_x_width' (area) by multiplying them.

        **Use case:** Captures how variables work together to affect outcomes.
        """,

        "Binning": """
        **Binning** transforms continuous numeric variables into categorical ones by grouping values into bins.

        **Example:** Age values (23, 35, 47, 52, 68) might be binned into categories like '20-40', '40-60', '60+'.

        **Methods:**
        - **Equal width:** Each bin has the same range of values
        - **Equal frequency:** Each bin has the same number of observations

        **Use case:** Reduces noise and improves model robustness.
        """,

        "Aggregation Features": """
        **Aggregation Features** calculate statistics for numeric variables grouped by categorical variables.

        **Example:** If you have sales data for different stores, you could create features like 'avg_sales_by_store_region' 
        or 'max_sales_by_product_category'.

        **Use case:** Captures group-level information that may influence individual observations.
        """,

        "Time Series Features": """
        **Time Series Features** extract patterns and properties from time-ordered data.

        **Rolling Window Features:** Calculate statistics over a moving time window (e.g., 7-day moving average)

        **Lag Features:** Use past values as features (e.g., yesterday's value, last week's value)

        **Use case:** Essential for forecasting and capturing temporal patterns.
        """,

        "Text Features": """
        **Text Features** extract measurable properties from text data.

        **Created features include:**
        - Text length (total characters)
        - Word count (number of words)
        - Capital ratio (proportion of uppercase letters)

        **Use case:** Converts unstructured text into structured features for analysis.
        """,

        "Cyclical Features": """
        **Cyclical Features** transform cyclic numeric variables (like hours, days, months) into continuous features that preserve their cyclical nature.

        **Problem it solves:** With regular encoding, December (12) and January (1) appear far apart numerically (12 vs 1), but they're actually adjacent months.

        **How it works:** 
        1. Converts each value to an angle: angle = 2π × value / max_value
        2. Creates two new features using sine and cosine:
           - column_sin = sin(angle)
           - column_cos = cos(angle)

        **Example with months:**
        | Month | Angle (radians) | month_sin | month_cos |
        |-------|----------------|-----------|-----------|
        | 1 (Jan) | 0.52 | 0.500 | 0.866 |
        | 12 (Dec) | 6.28 | 0.000 | 1.000 |

        Now December and January are close to each other in this new feature space!

        **Use case:** Improves models with time-based cyclic features (hours, weekdays, months, seasons).
        """
    }

    # Feature engineering options with expander for info
    fe_type = st.selectbox(
        "Select feature engineering technique:",
        ["Polynomial Features", "Interaction Features", "Binning", "Aggregation Features",
         "Time Series Features", "Text Features", "Cyclical Features"]
    )

    # Show explanation in an expander
    with st.expander("Learn more about this technique", expanded=False):
        if fe_type in feature_explanations:
            st.markdown(feature_explanations[fe_type])

    if fe_type == "Polynomial Features":
        if not numeric_columns:
            st.warning("No numeric columns available for polynomial features.")
            return None

        col = st.selectbox("Select column:", numeric_columns)
        degree = st.slider("Polynomial degree:", 2, 5, 2)

        # Preview
        with st.expander("Preview", expanded=True):
            result_df = fe_processor.create_polynomial_features(df, col, degree, record_to_pipeline=False)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            st.dataframe(result_df[new_cols].head(10), use_container_width=True)

        if st.button("Create Polynomial Features"):
            result = fe_processor.create_polynomial_features(df, col, degree, record_to_pipeline=True)
            st.success(f"Created {degree - 1} new polynomial features.")
            return result

    elif fe_type == "Interaction Features":
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for interaction features.")
            return None

        col1 = st.selectbox("First column:", numeric_columns)
        col2 = st.selectbox("Second column:",
                            [col for col in numeric_columns if col != col1])

        operation = st.selectbox(
            "Interaction type:",
            ["multiply", "divide", "add", "subtract"]
        )

        # Preview
        with st.expander("Preview", expanded=True):
            result_df = fe_processor.create_interaction_term(df, col1, col2, operation, record_to_pipeline=False)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            st.dataframe(result_df[new_cols].head(10), use_container_width=True)

        if st.button("Create Interaction Feature"):
            result = fe_processor.create_interaction_term(df, col1, col2, operation, record_to_pipeline=True)
            st.success(f"Created new interaction feature.")
            return result

    elif fe_type == "Binning":
        if not numeric_columns:
            st.warning("No numeric columns available for binning.")
            return None

        col = st.selectbox("Select column:", numeric_columns)
        method = st.radio("Binning method:", ["equal_width", "equal_frequency"])
        bins = st.slider("Number of bins:", 2, 10, 4)

        # Preview
        with st.expander("Preview", expanded=True):
            result_df = fe_processor.create_binned_feature(df, col, method, bins, record_to_pipeline=False)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            st.dataframe(result_df[new_cols].head(10), use_container_width=True)

        if st.button("Create Binned Feature"):
            result = fe_processor.create_binned_feature(df, col, method, bins, record_to_pipeline=True)
            st.success(f"Created new binned feature.")
            return result

    elif fe_type == "Aggregation Features":
        if not categorical_columns or not numeric_columns:
            st.warning("Need at least one categorical and one numeric column for aggregation features.")
            return None

        group_col = st.selectbox("Group by column:", categorical_columns)
        agg_col = st.selectbox("Column to aggregate:", numeric_columns)

        agg_options = ["mean", "median", "min", "max", "std"]
        agg_functions = st.multiselect("Aggregation functions:", agg_options, default=["mean"])

        if not agg_functions:
            st.warning("Please select at least one aggregation function.")
            return None

        # Preview
        with st.expander("Preview", expanded=True):
            result_df = fe_processor.create_aggregation_features(df, group_col, agg_col, agg_functions,
                                                                 record_to_pipeline=False)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            st.dataframe(result_df[new_cols].head(10), use_container_width=True)

        if st.button("Create Aggregation Features"):
            result = fe_processor.create_aggregation_features(df, group_col, agg_col, agg_functions,
                                                              record_to_pipeline=True)
            st.success(f"Created {len(agg_functions)} new aggregation features.")
            return result

    elif fe_type == "Time Series Features":
        if not datetime_columns or not numeric_columns:
            st.warning("Need at least one datetime and one numeric column for time series features.")
            return None

        date_col = st.selectbox("Date column:", datetime_columns)
        value_col = st.selectbox("Value column:", numeric_columns)

        ts_type = st.radio("Time series feature type:", ["Rolling Window", "Lag Features"])

        if ts_type == "Rolling Window":
            window_options = [3, 7, 14, 30, 90]
            windows = st.multiselect("Window sizes:", window_options, default=[7])

            function_options = ["mean", "std", "min", "max"]
            functions = st.multiselect("Functions:", function_options, default=["mean"])

            if not windows or not functions:
                st.warning("Please select at least one window size and function.")
                return None

            # Preview
            with st.expander("Preview", expanded=True):
                result_df = fe_processor.create_rolling_features(df, date_col, value_col, windows, functions,
                                                                 record_to_pipeline=False)
                new_cols = [c for c in result_df.columns if c not in df.columns]
                st.dataframe(result_df[new_cols].head(10), use_container_width=True)

            if st.button("Create Rolling Features"):
                result = fe_processor.create_rolling_features(df, date_col, value_col, windows, functions,
                                                              record_to_pipeline=True)
                st.success(f"Created {len(windows) * len(functions)} new rolling window features.")
                return result

        elif ts_type == "Lag Features":
            lag_options = [1, 3, 7, 14, 30]
            lags = st.multiselect("Lag periods:", lag_options, default=[1])

            if not lags:
                st.warning("Please select at least one lag period.")
                return None

            # Preview
            with st.expander("Preview", expanded=True):
                result_df = fe_processor.create_lag_features(df, date_col, value_col, lags, record_to_pipeline=False)
                new_cols = [c for c in result_df.columns if c not in df.columns]
                st.dataframe(result_df[new_cols].head(10), use_container_width=True)

            if st.button("Create Lag Features"):
                result = fe_processor.create_lag_features(df, date_col, value_col, lags, record_to_pipeline=True)
                st.success(f"Created {len(lags)} new lag features.")
                return result

    elif fe_type == "Text Features":
        text_columns = [col for col in categorical_columns
                        if df[col].dtype == 'object' and df[col].str.len().mean() > 10]

        if not text_columns:
            st.warning("No suitable text columns found.")
            return None

        text_col = st.selectbox("Text column:", text_columns)

        # Preview
        with st.expander("Preview", expanded=True):
            result_df = fe_processor.create_text_features(df, text_col, record_to_pipeline=False)
            new_cols = [c for c in result_df.columns if c not in df.columns]
            st.dataframe(result_df[new_cols].head(10), use_container_width=True)

        if st.button("Create Text Features"):
            result = fe_processor.create_text_features(df, text_col, record_to_pipeline=True)
            st.success(f"Created text features for {text_col}.")
            return result

    elif fe_type == "Cyclical Features":
        # Determine suitable columns for cyclical encoding
        cyclical_columns = []

        # Check for datetime components in column names
        for col in numeric_columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['hour', 'day', 'month', 'week', 'quarter', 'year', 'season']):
                cyclical_columns.append(col)

        # If no obvious cyclical columns, allow selection from all numeric columns
        if not cyclical_columns:
            cyclical_columns = numeric_columns

        if not cyclical_columns:
            st.warning("No numeric columns available for cyclical encoding.")
            return None

        col = st.selectbox("Select cyclical column:", cyclical_columns)

        # Try to determine max value based on column name
        max_value_suggestion = None
        col_lower = col.lower()
        if "hour" in col_lower:
            max_value_suggestion = 24
        elif "day" in col_lower and "week" in col_lower:
            max_value_suggestion = 7
        elif "month" in col_lower:
            max_value_suggestion = 12
        elif "day" in col_lower:
            max_value_suggestion = 31

        # Let user specify max value
        if max_value_suggestion:
            st.write(f"Detected cyclical feature: suggested maximum value is {max_value_suggestion}")
            use_suggestion = st.checkbox("Use suggested maximum value", value=True)
            if use_suggestion:
                max_value = max_value_suggestion
            else:
                max_value = st.number_input("Maximum value in cycle:", min_value=2, value=max_value_suggestion)
        else:
            max_value = st.number_input("Maximum value in cycle:", min_value=2, value=24)

        # Preview
        with st.expander("Preview", expanded=True):
            try:
                result_df = fe_processor.encode_cyclical_features(df, col, max_value, record_to_pipeline=False)
                new_cols = [c for c in result_df.columns if c not in df.columns]
                st.dataframe(result_df[new_cols].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error generating preview: {str(e)}")

        if st.button("Create Cyclical Features"):
            try:
                result = fe_processor.encode_cyclical_features(df, col, max_value, record_to_pipeline=True)
                st.success(f"Created cyclical features for {col}.")
                return result
            except Exception as e:
                st.error(f"Error creating cyclical features: {str(e)}")

    return None