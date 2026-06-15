import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.preprocessing_pipeline import record_step


class CategoricalProcessor(DataProcessor):
    """Class for processing categorical data"""

    def __init__(self):
        super().__init__()

    def get_frequency_stats(self, df, column):
        """Get frequency statistics for a categorical column"""
        value_counts = df[column].value_counts()
        most_common_value = value_counts.idxmax() if not value_counts.empty else None
        most_common = value_counts.max() if not value_counts.empty else 0

        return {
            'value_counts': value_counts,
            'most_common_value': most_common_value,
            'most_common': most_common
        }

    def get_unique_count(self, df, column):
        """Get number of unique values in a categorical column"""
        return df[column].nunique()

    @record_step("categorical")
    def apply_one_hot_encoding(self, df, column, keep_original=False):
        """Apply one-hot encoding to a categorical column"""
        result_df = df.copy()
        one_hot = pd.get_dummies(result_df[column], prefix=column)

        # Add the one-hot encoded columns
        for col in one_hot.columns:
            result_df[col] = one_hot[col]

        # Optionally remove the original column
        if not keep_original:
            result_df = result_df.drop(columns=[column])

        return result_df

    def get_label_mapping(self, df, column):
        """Get mapping for label encoding"""
        unique_values = sorted(df[column].dropna().unique())
        return {val: idx for idx, val in enumerate(unique_values)}

    @record_step("categorical")
    def apply_label_encoding(self, df, column, mapping=None):
        """Apply label encoding to a categorical column"""
        result_df = df.copy()

        if mapping is None:
            mapping = self.get_label_mapping(df, column)

        result_df[f"{column}_encoded"] = result_df[column].map(mapping)
        return result_df

    @record_step("categorical")
    def merge_categories(self, df, column, categories_to_merge, new_category_name):
        """Merge multiple categories into one"""
        result_df = df.copy()
        new_values = result_df[column].copy()
        mask = new_values.isin(categories_to_merge)
        new_values[mask] = new_category_name
        result_df[f"{column}_merged"] = new_values
        return result_df