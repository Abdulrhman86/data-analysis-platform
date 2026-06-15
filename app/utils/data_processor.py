import pandas as pd
import numpy as np
from utils.preprocessing_pipeline import record_step


class DataProcessor:
    """Base class for data processing operations"""

    def __init__(self):
        pass

    def get_missing_count(self, df, column):
        """Return the number of missing values in a column"""
        return df[column].isna().sum()

    @record_step("data")
    def replace_missing_values(self, df, column, value, inplace=True):
        """
        Replace missing values in a column

        Args:
            df: pandas DataFrame
            column: column to process
            value: value to use for replacement
            inplace: if True, replace values in original column; if False, create a new column

        Returns:
            DataFrame with replaced missing values
        """
        result_df = df.copy()

        if inplace:
            # Replace missing values in the original column
            result_df[column] = df[column].fillna(value)
        else:
            # Create a new column with "_filled" suffix
            result_df[f"{column}_filled"] = df[column].fillna(value)

        return result_df

    def get_replacement_value(self, df, column, strategy, custom_value=None):
        """
        Get replacement value for missing values based on strategy

        Args:
            df: DataFrame containing the data
            column: Column name to analyze
            strategy: Strategy to use ('mean', 'median', 'mode', 'zero', 'custom', 'most_frequent')
            custom_value: Custom value to use if strategy is 'custom'

        Returns:
            Appropriate replacement value based on strategy
        """
        # For numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            if strategy.lower() == 'mean':
                return df[column].mean()
            elif strategy.lower() == 'median':
                return df[column].median()
            elif strategy.lower() == 'mode':
                mode_result = df[column].mode()
                return mode_result[0] if not mode_result.empty else 0
            elif strategy.lower() == 'zero':
                return 0
            elif strategy.lower() == 'custom':
                return custom_value
        # For categorical columns
        else:
            if strategy.lower() == 'most_frequent' or strategy.lower() == 'mode':
                mode_result = df[column].mode()
                return mode_result[0] if not mode_result.empty else "Unknown"
            elif strategy.lower() == 'custom':
                return custom_value

        return None  # If no valid strategy
