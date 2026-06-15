import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.preprocessing_pipeline import record_step


class DatetimeProcessor(DataProcessor):
    """Class for processing datetime data"""

    def __init__(self):
        super().__init__()

    @record_step("datetime")
    def extract_components(self, df, column):
        """Extract year, month, day, weekday from datetime column"""
        result_df = df.copy()
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(df[column]):
            result_df[column] = pd.to_datetime(df[column], errors='coerce')

        # Extract components
        result_df[f"{column}_year"] = result_df[column].dt.year
        result_df[f"{column}_month"] = result_df[column].dt.month
        result_df[f"{column}_day"] = result_df[column].dt.day
        result_df[f"{column}_weekday"] = result_df[column].dt.weekday
        result_df[f"{column}_quarter"] = result_df[column].dt.quarter

        return result_df

    @record_step("datetime")
    def convert_to_time_since(self, df, column, reference_date=None):
        """Convert datetime to time since reference date"""
        result_df = df.copy()
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(df[column]):
            result_df[column] = pd.to_datetime(df[column], errors='coerce')

        # If no reference date, use minimum date in the column
        if reference_date is None:
            reference_date = result_df[column].min()
        else:
            reference_date = pd.to_datetime(reference_date)

        # Calculate time difference in days
        result_df[f"{column}_days_since"] = (result_df[column] - reference_date).dt.total_seconds() / (24 * 60 * 60)

        return result_df

    @record_step("datetime")
    def replace_missing_values(self, df, column, value=None, method='fill_na', inplace=True):
        """
        Replace missing values in a datetime column

        Args:
            df: pandas DataFrame
            column: column to process
            value: value to use for replacement (datetime object or string)
            method: method to use ('fill_na', 'forward_fill', 'backward_fill', 'interpolate')
            inplace: if True, replace values in original column; if False, create a new column

        Returns:
            DataFrame with replaced missing values
        """
        result_df = df.copy()

        # Ensure column is datetime type
        if not pd.api.types.is_datetime64_dtype(result_df[column]):
            try:
                result_df[column] = pd.to_datetime(result_df[column], errors='coerce')
            except Exception as e:
                raise ValueError(f"Could not convert column {column} to datetime: {str(e)}")

        # Create a copy of the column to modify
        if inplace:
            col_to_fill = result_df[column]
        else:
            result_df[f"{column}_filled"] = result_df[column].copy()
            col_to_fill = result_df[f"{column}_filled"]

        # Apply the specified method
        if method == 'fill_na':
            if value is None:
                raise ValueError("Value must be provided when using 'fill_na' method")

            # Convert value to datetime if it's not already
            if not isinstance(value, pd.Timestamp):
                try:
                    value = pd.to_datetime(value)
                except:
                    raise ValueError(f"Could not convert '{value}' to datetime")

            # Apply the fill
            if inplace:
                result_df[column] = col_to_fill.fillna(value)
            else:
                result_df[f"{column}_filled"] = col_to_fill.fillna(value)

        elif method == 'forward_fill':
            if inplace:
                result_df[column] = col_to_fill.ffill()
            else:
                result_df[f"{column}_filled"] = col_to_fill.ffill()

        elif method == 'backward_fill':
            if inplace:
                result_df[column] = col_to_fill.bfill()
            else:
                result_df[f"{column}_filled"] = col_to_fill.bfill()

        else:
            raise ValueError(
                f"Unknown method: {method}. Supported methods are 'fill_na', 'forward_fill', 'backward_fill', 'interpolate'")

        return result_df

    def get_replacement_value(self, df, column, strategy, custom_value=None):
        """
        Get replacement value for missing datetime values based on strategy

        Args:
            df: DataFrame containing the data
            column: Column name to analyze
            strategy: Strategy to use ('min', 'max', 'median', 'mode', 'custom')
            custom_value: Custom value to use if strategy is 'custom'

        Returns:
            Appropriate replacement value based on strategy
        """
        # Convert to datetime if needed
        temp_col = df[column]
        if not pd.api.types.is_datetime64_dtype(temp_col):
            temp_col = pd.to_datetime(temp_col, errors='coerce')

        # Get non-null values
        non_null = temp_col.dropna()

        if non_null.empty:
            # No valid values to compute from, return current datetime
            return pd.Timestamp.now()

        if strategy.lower() == 'min':
            return non_null.min()
        elif strategy.lower() == 'max':
            return non_null.max()
        elif strategy.lower() == 'median':
            return non_null.sort_values().iloc[len(non_null) // 2]
        elif strategy.lower() == 'mode':
            mode_result = non_null.mode()
            return mode_result[0] if not mode_result.empty else non_null.min()
        elif strategy.lower() == 'custom':
            if custom_value is None:
                return pd.Timestamp.now()

            # Try to convert custom value to datetime
            try:
                return pd.to_datetime(custom_value)
            except:
                # If conversion fails, return current date
                return pd.Timestamp.now()

        # Default to minimum date if strategy not recognized
        return non_null.min()