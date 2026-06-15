import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Assessing data quality...")
def enhanced_data_quality(df):
    """
    Comprehensive data quality assessment with preprocessing recommendations

    Args:
        df: pandas DataFrame

    Returns:
        dict: Dictionary with quality metrics and recommendations
    """
    from scipy import stats
    import numpy as np

    # Initialize quality assessment dictionary
    quality = {
        'summary': {
            'row_count': len(df),
            'column_count': len(df.columns),
            'duplicated_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        },
        'columns': {},
        'recommendations': []
    }

    # Detect column types
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns

    # Process each column
    for col in df.columns:
        col_data = {
            'missing_count': df[col].isna().sum(),
            'missing_percentage': round(df[col].isna().sum() / len(df) * 100, 2),
            'unique_count': df[col].nunique(),
            'unique_percentage': round(df[col].nunique() / len(df) * 100, 2),
        }

        # Add column-specific stats and recommendations
        if col in numeric_cols:
            col_data['type'] = 'numeric'
            col_data['min'] = df[col].min()
            col_data['max'] = df[col].max()
            col_data['mean'] = df[col].mean()
            col_data['median'] = df[col].median()
            col_data['std'] = df[col].std()

            # Check for skewness
            skewness = df[col].skew()
            col_data['skewness'] = skewness

            # Check for outliers using IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            col_data['outliers_count'] = outliers
            col_data['outliers_percentage'] = round(outliers / df[col].count() * 100, 2)

            # Add recommendations
            if outliers > 0 and outliers / df[col].count() > 0.05:
                quality['recommendations'].append({
                    'column': col,
                    'issue': 'outliers',
                    'description': f'Column has {outliers} outliers ({col_data["outliers_percentage"]}%). Consider using "Handle Outliers" functionality.',
                    'severity': 'medium' if col_data["outliers_percentage"] > 10 else 'low'
                })

            if abs(skewness) > 1:
                quality['recommendations'].append({
                    'column': col,
                    'issue': 'skewness',
                    'description': f'Column is {"positively" if skewness > 0 else "negatively"} skewed ({skewness:.2f}). Consider a log transform or other normalization.',
                    'severity': 'medium' if abs(skewness) > 2 else 'low'
                })

            # Check for constant or near-constant values
            if col_data['unique_count'] == 1:
                quality['recommendations'].append({
                    'column': col,
                    'issue': 'constant',
                    'description': f'Column has only one unique value. Consider removing it as it provides no information.',
                    'severity': 'high'
                })

        elif col in categorical_cols:
            col_data['type'] = 'categorical'
            col_data['most_common'] = df[col].value_counts().index[0] if not df[col].empty else None
            col_data['most_common_count'] = df[col].value_counts().iloc[0] if not df[col].empty else 0
            col_data['most_common_percentage'] = round(col_data['most_common_count'] / len(df) * 100, 2)

            # Check for high cardinality
            if col_data['unique_count'] > 50 and col_data['unique_percentage'] > 25:
                quality['recommendations'].append({
                    'column': col,
                    'issue': 'high_cardinality',
                    'description': f'Column has {col_data["unique_count"]} unique values ({col_data["unique_percentage"]}%). One-hot encoding may create too many features. Consider label encoding or target encoding.',
                    'severity': 'medium'
                })

            # Check for imbalanced categories
            if col_data['most_common_percentage'] > 90:
                quality['recommendations'].append({
                    'column': col,
                    'issue': 'imbalanced',
                    'description': f'Column is highly imbalanced. Most common value "{col_data["most_common"]}" appears in {col_data["most_common_percentage"]}% of rows.',
                    'severity': 'medium'
                })

            # Check for potential numeric data stored as strings
            if col_data['unique_count'] > 10:
                # Try to convert to numeric
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                successful_conversion = (~numeric_conversion.isna()).mean() > 0.8  # 80% conversion success

                if successful_conversion:
                    quality['recommendations'].append({
                        'column': col,
                        'issue': 'type_conversion',
                        'description': f'Column may contain numeric data stored as strings. Consider converting to numeric.',
                        'severity': 'medium'
                    })

            # Check for potential datetime data
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='raise')
                    quality['recommendations'].append({
                        'column': col,
                        'issue': 'type_conversion',
                        'description': f'Column contains date/time data. Consider converting to datetime format.',
                        'severity': 'medium'
                    })
                except:
                    pass

        elif col in datetime_cols:
            col_data['type'] = 'datetime'
            col_data['min'] = df[col].min()
            col_data['max'] = df[col].max()
            col_data['range_days'] = (df[col].max() - df[col].min()).days

            # Recommend feature extraction
            quality['recommendations'].append({
                'column': col,
                'issue': 'feature_extraction',
                'description': f'Consider extracting year, month, day, weekday from this datetime column to create useful features.',
                'severity': 'low'
            })

        # Common checks for all column types
        if col_data['missing_percentage'] > 0:
            severity = 'low' if col_data['missing_percentage'] < 5 else (
                'medium' if col_data['missing_percentage'] < 20 else 'high')
            quality['recommendations'].append({
                'column': col,
                'issue': 'missing_values',
                'description': f'Column has {col_data["missing_percentage"]}% missing values. Consider imputation or removal.',
                'severity': severity
            })

        quality['columns'][col] = col_data

    # Dataset-level recommendations
    if quality['summary']['duplicated_rows'] > 0:
        quality['recommendations'].append({
            'column': None,
            'issue': 'duplicated_rows',
            'description': f'Dataset has {quality["summary"]["duplicated_rows"]} duplicated rows ({round(quality["summary"]["duplicated_rows"] / len(df) * 100, 2)}%). Consider removing duplicates.',
            'severity': 'medium'
        })

    # Sort recommendations by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    quality['recommendations'] = sorted(quality['recommendations'], key=lambda x: severity_order[x['severity']])

    return quality