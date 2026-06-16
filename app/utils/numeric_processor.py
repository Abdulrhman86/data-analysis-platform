import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.preprocessing_pipeline import record_step

class NumericProcessor(DataProcessor):
    """Class for processing numeric data"""

    def __init__(self):
        super().__init__()

    def get_basic_statistics(self, df, column):
        """Calculate basic statistics for a numeric column"""
        stats = {
            'count': df[column].count(),
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            '25%': df[column].quantile(0.25),
            'median': df[column].median(),
            '75%': df[column].quantile(0.75),
            'max': df[column].max(),
            'skewness': df[column].skew(),
            'kurtosis': df[column].kurt()
        }
        return pd.DataFrame([stats]).T

    @record_step("numeric")
    def normalize(self, df, column, add_as_new_column=True):
        """
        Normalize a column to 0-1 range

        Args:
            df: pandas DataFrame
            column: column to normalize
            add_as_new_column: if True, adds result as a new column; if False, returns the values

        Returns:
            DataFrame with new column or normalized values depending on add_as_new_column
        """
        min_val = df[column].min()
        max_val = df[column].max()
        range_val = max_val - min_val

        if range_val == 0:
            # Can't normalize, all values are the same
            return df if add_as_new_column else df[column]

        normalized_values = (df[column] - min_val) / range_val

        if add_as_new_column:
            result_df = df.copy()
            result_df[f"{column}_normalized"] = normalized_values
            return result_df
        else:
            return normalized_values

    @record_step("numeric")
    def standardize(self, df, column, add_as_new_column=True):
        """
        Standardize a column (z-score)

        Args:
            df: pandas DataFrame
            column: column to standardize
            add_as_new_column: if True, adds result as a new column; if False, returns the values

        Returns:
            DataFrame with new column or standardized values depending on add_as_new_column
        """
        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            # Can't standardize, standard deviation is zero
            return df if add_as_new_column else df[column]

        standardized_values = (df[column] - mean) / std

        if add_as_new_column:
            result_df = df.copy()
            result_df[f"{column}_standardized"] = standardized_values
            return result_df
        else:
            return standardized_values


    def get_bin_edges(self, df, column, bins):
        """Get bin edges for binning"""
        return np.linspace(df[column].min(), df[column].max(), bins + 1)

    @record_step("numeric")
    def apply_binning(self, df, column, bins, method='Equal width'):
        """Apply binning to a column"""
        result_df = df.copy()
        if method == 'Equal width':
            result_df[f"{column}_binned"] = pd.cut(df[column], bins)
        else:  # Equal frequency
            result_df[f"{column}_binned"] = pd.qcut(df[column], bins, duplicates='drop')
        return result_df

    @record_step("numeric")
    def apply_log_transform(self, df, column, offset=0):
        """Apply a log transform to a column.

        Non-positive inputs make ``np.log`` return -inf (zeros) or NaN (negatives),
        which silently corrupts the feature when a recorded recipe is replayed on a
        new dataset whose values dip to/below zero. We clip to a tiny positive value
        so the output is always finite (NaNs in the input are preserved)."""
        result_df = df.copy()
        col = pd.to_numeric(df[column], errors='coerce') + offset
        result_df[f"{column}_log"] = np.log(col.clip(lower=1e-9))
        return result_df

    def calculate_outlier_bounds(self, df, column):
        """Calculate outlier bounds using IQR method"""
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def detect_outliers(self, df, column, lower_bound, upper_bound):
        """Detect outliers in a column"""
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].tolist()

    @record_step("numeric")
    def handle_outliers(self, df, column, lower_bound, upper_bound, method='Cap at bounds'):
        """Handle outliers using specified method"""
        result_df = df.copy()

        if method == 'Cap at bounds':
            result_df[f"{column}_capped"] = df[column].clip(lower=lower_bound, upper=upper_bound)
            return result_df
        elif method == 'Remove outliers':
            return result_df[(result_df[column] >= lower_bound) & (result_df[column] <= upper_bound)]
        else:  # Replace with NaN
            column_copy = result_df[column].copy()
            outlier_mask = (result_df[column] < lower_bound) | (result_df[column] > upper_bound)
            column_copy[outlier_mask] = np.nan
            result_df[f"{column}_no_outliers"] = column_copy
            return result_df