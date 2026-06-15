import pandas as pd
import numpy as np
import re
from datetime import datetime

def assess_data_quality(df):
    """
    Perform a comprehensive data quality assessment

    Args:
        df: pandas DataFrame

    Returns:
        dict: Dictionary with data quality metrics
    """
    # Initialize result dictionary
    quality = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': {},
        'duplicates': df.duplicated().sum(),
        'column_stats': {}
    }

    # Check for missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            quality['missing_values'][col] = {
                'count': missing,
                'percentage': round(missing / len(df) * 100, 2)
            }

    # Column-wise statistics
    for col in df.columns:
        col_stats = {
            'unique_values': df[col].nunique(),
            'unique_percentage': round(df[col].nunique() / len(df) * 100, 2)
        }

        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            })

        quality['column_stats'][col] = col_stats

    return quality


def try_convert_date_columns(df):
    """
    Try to convert columns that look like dates into datetime format.
    Uses both column name analysis and data pattern recognition to identify date columns.

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame with converted datetime columns
    """
    result_df = df.copy()

    # Date-related keywords for column name checking
    date_keywords = ['date', 'day', 'month', 'year', 'time', 'dt', 'created', 'updated',
                     'modified', 'birth', 'start', 'end', 'timestamp']

    # Common date patterns for regex checking
    date_patterns = [
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # DD/MM/YYYY, MM/DD/YYYY
        r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',  # YYYY/MM/DD, YYYY-MM-DD
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s\d{1,2}:\d{1,2}(:\d{1,2})?$',  # DD/MM/YYYY HH:MM(:SS)
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}\s\d{1,2}:\d{1,2}(:\d{1,2})?$',  # YYYY/MM/DD HH:MM(:SS)
        r'^\d{1,2}[-\s]\w{3,9}[-\s]\d{2,4}$',  # DD Mon YYYY, DD Month YYYY
        r'^\w{3,9}[-\s]\d{1,2}[-\s]\d{2,4}$',  # Mon DD YYYY, Month DD YYYY
        r'^\d{8}$',  # YYYYMMDD, DDMMYYYY
        r'^\d{1,2}:\d{1,2}(:\d{1,2})?(\s[AP]M)?$',  # HH:MM(:SS) (AM/PM)
        r'^[A-Za-z]{3}\s\d{1,2},?\s\d{4}$'  # Jan 1, 2020
    ]

    # Date formats to try (corresponding to some of the patterns above)
    date_formats = [
        '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%Y-%m-%d',
        '%d-%m-%Y', '%m-%d-%Y',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
        '%d %b %Y', '%d %B %Y',
        '%b %d %Y', '%B %d %Y',
        '%Y%m%d', '%d%m%Y',
        '%H:%M:%S', '%I:%M:%S %p',
        '%b %d, %Y'
    ]

    for col in result_df.columns:
        # Skip columns that are already datetime type
        if pd.api.types.is_datetime64_dtype(result_df[col]):
            continue

        # Skip numeric columns with large values that are unlikely to be dates
        if pd.api.types.is_numeric_dtype(result_df[col]):
            if result_df[col].dropna().min() > 20000000:  # Arbitrary large number
                continue

        # Method 1: Check if column name contains date-related keywords
        name_suggests_date = any(keyword in col.lower() for keyword in date_keywords)

        # Method 2: Check if the data matches common date patterns
        data_looks_like_date = False
        pattern_matches = 0

        # Only check string columns or columns that could be converted to strings
        if result_df[col].dtype == 'object' or pd.api.types.is_numeric_dtype(result_df[col]):
            # Sample up to 100 non-null values
            sample = result_df[col].dropna().astype(str).head(100)

            # If there are values in the sample
            if not sample.empty:
                # Check if sample values match date patterns
                for pattern in date_patterns:
                    # If more than 60% of sampled values match a date pattern, it's likely a date column
                    matches = sample.str.match(pattern).mean()
                    if matches > 0.6:  # 60% threshold
                        data_looks_like_date = True
                        break

        # Proceed with conversion attempts if either method suggests a date column
        if name_suggests_date or data_looks_like_date:
            # Try automatic conversion first
            try:
                converted = pd.to_datetime(result_df[col], errors='coerce')

                # If more than 50% values converted successfully, it's likely a date column
                success_rate = 1 - (converted.isna().sum() / len(converted))
                if success_rate >= 0.5:
                    result_df[col] = converted
                    continue  # Column converted successfully, move to next column
            except Exception:
                pass  # If automatic conversion fails, try specific formats

            # Try specific date formats
            best_format = None
            best_success_rate = 0

            for fmt in date_formats:
                try:
                    converted = pd.to_datetime(result_df[col].astype(str), format=fmt, errors='coerce')
                    success_rate = 1 - (converted.isna().sum() / len(converted))

                    # Keep track of the format with the highest success rate
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_format = fmt
                except Exception:
                    continue

            # Apply the best format if it has a good success rate
            if best_format is not None and best_success_rate >= 0.5:
                try:
                    result_df[col] = pd.to_datetime(result_df[col].astype(str), format=best_format, errors='coerce')
                except Exception:
                    # Last resort: try automatic conversion again with errors='coerce'
                    converted = pd.to_datetime(result_df[col], errors='coerce')
                    if converted.isna().sum() <= 0.5 * len(converted):
                        result_df[col] = converted

    return result_df


# Utility function to check if Excel date numbers are present
def is_excel_date(series):
    """
    Check if a series contains Excel date numbers (integers around 40000-45000)
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False

    # Sample values
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False

    # Check if values are in typical Excel date range (roughly 1900-2100)
    # Excel dates start at 1 for Jan 1, 1900
    min_date = 1  # Jan 1, 1900
    max_date = 73500  # Roughly year 2100

    return (
            (sample >= min_date) & (sample <= max_date)
    ).mean() >= 0.8  # 80% of values in range


def detect_column_types(df):
    """
    Detect the data type of each column in the dataframe

    Args:
        df: pandas DataFrame

    Returns:
        dict: Dictionary with column names as keys and data types as values
    """
    column_types = {}

    for col in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'numeric'
        # Check if the column is already datetime
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types[col] = 'datetime'
        # Check if the column is boolean
        elif pd.api.types.is_bool_dtype(df[col]):
            column_types[col] = 'boolean'
        # Otherwise consider it categorical
        else:
            # Check for date patterns (more comprehensive check)
            if df[col].dtype == 'object':  # Only check string columns
                # Sample the first 100 non-null values
                sample = df[col].dropna().head(100).astype(str)

                # Common date indicators in column name
                date_keywords = ['date', 'day', 'month', 'year', 'time', 'birth', 'created', 'modified']
                has_date_keyword = any(keyword in col.lower() for keyword in date_keywords)

                # Check for date-like patterns in the data
                date_patterns = [
                    r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
                    r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY or DD-MM-YYYY
                    r'\d{4}/\d{1,2}/\d{1,2}',    # YYYY/MM/DD
                    r'\d{4}-\d{1,2}-\d{1,2}'     # YYYY-MM-DD
                ]

                looks_like_date = False
                for pattern in date_patterns:
                    if sample.str.match(pattern).any():
                        looks_like_date = True
                        break

                if has_date_keyword or looks_like_date:
                    # Try parsing as date
                    try:
                        # First try automatic conversion
                        pd.to_datetime(df[col], errors='raise')
                        column_types[col] = 'datetime'
                        continue
                    except Exception:
                        # Try with specific format for MM/DD/YYYY
                        try:
                            pd.to_datetime(df[col], format='%m/%d/%Y', errors='raise')
                            column_types[col] = 'datetime'
                            continue
                        except Exception:
                            pass

            # Check if it might be a numeric column stored as string
            try:
                if df[col].str.match(r'^[-+]?[0-9]*\.?[0-9]+$').all():
                    column_types[col] = 'numeric_as_string'
                else:
                    column_types[col] = 'categorical'
            except Exception:
                column_types[col] = 'categorical'

    return column_types


def suggest_preprocessing(column_types):
    """
    Suggest preprocessing methods based on column types

    Args:
        column_types: Dictionary with column names as keys and data types as values

    Returns:
        dict: Dictionary with suggested preprocessing methods for each column
    """
    suggestions = {}

    for col, col_type in column_types.items():
        if col_type == 'numeric':
            suggestions[col] = ['Normalize', 'Standardize', 'Log Transform', 'Binning']
        elif col_type == 'categorical':
            suggestions[col] = ['One-Hot Encode', 'Label Encode']
        elif col_type == 'datetime':
            suggestions[col] = ['Extract year/month/day', 'Convert to time since epoch']
        elif col_type == 'numeric_as_string':
            suggestions[col] = ['Convert to numeric', 'Normalize', 'Standardize']
        elif col_type == 'boolean':
            suggestions[col] = ['Convert to 0/1']

    return suggestions


def identify_columns_with_missing_values(df):
    """
    Identify columns with missing values

    Args:
        df: pandas DataFrame

    Returns:
        dict: Dictionary with column names as keys and number of missing values as values
    """
    missing_values = {}

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_values[col] = missing_count

    return missing_values


def identify_columns_with_outliers(df, column_types):
    """
    Identify numeric columns with potential outliers using IQR method

    Args:
        df: pandas DataFrame
        column_types: Dictionary with column names as keys and data types as values

    Returns:
        dict: Dictionary with column names as keys and outlier counts as values
    """
    outliers = {}

    for col, col_type in column_types.items():
        if col_type == 'numeric':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outliers[col] = outlier_count

    return outliers