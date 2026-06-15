import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.numeric_processor import NumericProcessor
from utils.categorical_processor import CategoricalProcessor
from utils.ui_helpers import create_transform_preview, batch_process_columns
from utils.preprocessing_utils import detect_column_types
from utils.datetime_processor import DatetimeProcessor
from utils.feature_engineering import feature_engineering_ui, FeatureEngineeringProcessor
from utils.preprocessing_pipeline import pipeline_manager_ui
from config import Config


def manage_rows_columns(df, missing_columns):
    """
    Unified management of rows and columns with missing values

    Args:
        df: pandas DataFrame
        missing_columns: Series of columns with missing values

    Returns:
        DataFrame: Modified DataFrame if changes were made, None otherwise
    """
    modified_df = None

    # Create tabs for different management options
    management_tabs = st.tabs(["Manage Rows", "Manage Columns"])

    with management_tabs[0]:  # Manage Rows
        st.markdown('<div class="tab-subheader">Manage Rows with Missing Values</div>', unsafe_allow_html=True)

        # Let user select which column's missing values to view
        selected_column = st.selectbox(
            "Select column with missing values:",
            options=missing_columns.index.tolist(),
            key="manage_rows_column_select"
        )

        # Get rows with missing values in the selected column
        missing_rows = df[df[selected_column].isnull()]

        st.write(f"Found {len(missing_rows)} rows with missing values in column '{selected_column}'")

        # Display the rows with missing values
        if not missing_rows.empty:
            st.dataframe(missing_rows.head(10), use_container_width=True)
            if len(missing_rows) > 10:
                st.markdown(
                    "<div style='text-align: right; color: #9CA3AF; font-size: 0.8rem;'>Showing first 10 rows</div>",
                    unsafe_allow_html=True)

        # Actions for row management
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Delete Selected Rows")
            # Create a list of row indices for selection
            row_indices = missing_rows.index.tolist()
            selected_row_indices = st.multiselect(
                "Select rows to delete:",
                options=row_indices,
                format_func=lambda x: f"Row {x}",
                key="selected_row_indices"
            )

            if selected_row_indices and st.button("Delete Selected Rows", key="delete_selected_btn",
                                                  use_container_width=True):
                # Create a copy of the original dataframe without the selected rows
                modified_df = df.copy()
                modified_df = modified_df.drop(selected_row_indices)
                st.success(f"Successfully deleted {len(selected_row_indices)} rows from the dataset.")

        with col2:
            st.markdown("#### Delete All Missing Rows")
            if st.button(f"Delete ALL rows with missing values in '{selected_column}'", key="delete_all_missing_btn",
                         use_container_width=True):
                modified_df = df.copy()
                # Get rows without missing values in the selected column
                modified_df = modified_df.dropna(subset=[selected_column])
                rows_removed = len(df) - len(modified_df)
                st.success(f"Successfully deleted {rows_removed} rows with missing values in '{selected_column}'.")

    with management_tabs[1]:  # Manage Columns
        st.markdown('<div class="tab-subheader">Manage Columns with Missing Values</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Multi-select for columns with missing values to delete
            st.markdown("#### Delete Columns with Missing Values")
            missing_cols = missing_columns.index.tolist()
            missing_cols_to_delete = st.multiselect(
                "Select columns with missing values to delete:",
                options=missing_cols,
                key="missing_cols_delete"
            )

            if missing_cols_to_delete and st.button("Delete Selected Columns", key="delete_missing_cols_btn",
                                                    use_container_width=True):
                modified_df = df.copy()
                modified_df = modified_df.drop(columns=missing_cols_to_delete)
                st.success(
                    f"Successfully deleted {len(missing_cols_to_delete)} columns: {', '.join(missing_cols_to_delete)}")

        with col2:
            # General column management
            st.markdown("#### Delete Any Columns")
            columns_to_delete = st.multiselect(
                "Select any columns to delete:",
                options=df.columns.tolist(),
                key="any_cols_delete"
            )

            if columns_to_delete and st.button("Delete Selected Columns", key="delete_any_cols_btn",
                                               use_container_width=True):
                modified_df = df.copy()
                modified_df = modified_df.drop(columns=columns_to_delete)
                st.success(f"Successfully deleted {len(columns_to_delete)} columns: {', '.join(columns_to_delete)}")

    # Return the modified dataframe if changes were made
    return modified_df

def create_missing_values_ui(st, df, column, processor, is_numeric=True):
    """
    Create a unified UI for handling missing values that works for both numeric and categorical columns

    Args:
        st: Streamlit instance
        df: DataFrame
        column: Column name
        processor: The appropriate processor (NumericProcessor or CategoricalProcessor)
        is_numeric: Whether the column is numeric (determines replacement options)

    Returns:
        Modified DataFrame if changes are applied, otherwise None
    """
    missing_count = processor.get_missing_count(df, column)

    if missing_count > 0:
        st.write(
            f"Column '{column}' has {missing_count} missing values ({(missing_count / len(df) * 100):.2f}% of total rows).")

        # Add option to view rows with missing values
        if st.checkbox(f"View rows with missing values in '{column}'"):
            missing_rows = df[df[column].isna()]

            # Provide context about these rows in an expander
            with st.expander("Missing values context", expanded=False):
                st.write("Profile of rows with missing values:")

                # Get a list of other columns to analyze
                other_cols = [col for col in df.columns if col != column]

                if other_cols:
                    # Create a single dataframe with distribution stats for categorical columns
                    dist_data = []

                    for col in other_cols:
                        if df[col].dtype == 'object' or df[col].nunique() < 10:
                            # Get value counts for this column in rows with missing values
                            value_counts = missing_rows[col].value_counts()

                            # Convert to percentage
                            total = value_counts.sum()
                            if total > 0:  # Avoid division by zero
                                for val, count in value_counts.items():
                                    dist_data.append({
                                        'Column': col,
                                        'Value': val,
                                        'Count': count,
                                        'Percentage': f"{(count / total * 100):.1f}%"
                                    })

                    if dist_data:
                        dist_df = pd.DataFrame(dist_data)
                        st.write("Distribution of values in other columns:")
                        st.dataframe(dist_df, use_container_width=True)
                    else:
                        st.write("No categorical columns found for context analysis.")

            # Show the missing rows
            st.write(f"Showing {len(missing_rows)} rows with missing values in column '{column}':")
            st.dataframe(missing_rows, use_container_width=True)

        # Replacement strategy options
        st.subheader("Replace Missing Values")

        # Initialize value to avoid None
        value = None

        if is_numeric:
            strategy = st.selectbox(
                "Replacement strategy:",
                ["Mean", "Median", "Mode", "Zero", "Custom Value"],
                key=f"strategy_{column}"
            )

            if strategy == "Custom Value":
                custom_value = st.number_input("Replacement value:", key=f"custom_value_{column}")
                value = processor.get_replacement_value(df, column, 'custom', custom_value)
            else:
                value = processor.get_replacement_value(df, column, strategy.lower())
                st.write(f"{strategy} value: {value}")
        else:
            # For categorical columns - simplified options
            strategy = st.selectbox(
                "Replacement strategy:",
                ["Most Frequent Value", "Custom Value"],
                key=f"strategy_{column}"
            )

            if strategy == "Most Frequent Value":
                value = processor.get_replacement_value(df, column, 'most_frequent')
                st.write(f"Most frequent value: '{value}'")
            else:  # Custom Value
                custom_value = st.text_input("Enter custom value to use for missing entries:", "Unknown",
                                             key=f"custom_value_{column}")
                value = processor.get_replacement_value(df, column, 'custom', custom_value)
                st.write("This will replace all missing values with your custom text.")

        # Option to replace in original column or create new column
        replacement_mode = st.radio(
            "How to replace missing values:",
            ["Replace in original column", "Create new column with filled values"],
            index=0,
            key=f"replace_mode_{column}"
        )
        inplace = replacement_mode == "Replace in original column"

        # Preview data with filled values
        if value is not None:  # Ensure value is not None
            with st.expander("Preview of data with filled values", expanded=True):
                preview_df = df.copy()

                # Create a mask for missing values to highlight them
                missing_mask = preview_df[column].isna()

                if inplace:
                    preview_df[column] = preview_df[column].fillna(value)
                else:
                    new_col_name = f"{column}_filled"
                    preview_df[new_col_name] = preview_df[column].fillna(value)

                # Get the rows that had missing values
                missing_rows_preview = df[missing_mask].copy()

                if not missing_rows_preview.empty:
                    # Show the filled values in those specific rows
                    if inplace:
                        # For inplace, we need to show the filled rows
                        filled_rows = preview_df.loc[missing_rows_preview.index]
                        st.write(f"Preview of rows that had missing values (now filled with '{value}'):")
                        st.dataframe(filled_rows.head(10), use_container_width=True)
                    else:
                        # For new column, show both original and new column
                        filled_rows = preview_df.loc[missing_rows_preview.index]
                        display_cols = [column, new_col_name]
                        st.write(f"Preview of rows with missing values (showing original and filled columns):")
                        st.dataframe(filled_rows[display_cols].head(10), use_container_width=True)
                else:
                    st.write("No missing values found in this preview sample.")

                # Show some statistics about the fill
                st.write(f"Total missing values that will be filled: {missing_count}")

                if is_numeric:
                    # For numeric columns, show before/after statistics
                    stats_before = df[column].describe()
                    if inplace:
                        stats_after = preview_df[column].describe()
                    else:
                        stats_after = preview_df[f"{column}_filled"].describe()

                    # Show side by side statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Statistics before filling:")
                        st.write(stats_before)
                    with col2:
                        st.write("Statistics after filling:")
                        st.write(stats_after)

            # Button to apply changes
            if st.button(f"Replace Missing Values in {column}", key=f"apply_btn_{column}"):
                if inplace:
                    # Use the processor method with record_to_pipeline=True
                    result_df = processor.replace_missing_values(
                        df,
                        column,
                        value,
                        inplace=True,
                        record_to_pipeline=True
                    )
                    st.success(f"Replaced {missing_count} missing values in column: {column}")
                else:
                    # Use the processor method with record_to_pipeline=True
                    result_df = processor.replace_missing_values(
                        df,
                        column,
                        value,
                        inplace=False,
                        record_to_pipeline=True
                    )
                    st.success(f"Created new column '{column}_filled' with filled values")

                return result_df


        else:
            st.warning("Please select a valid replacement strategy and value.")
    else:
        st.info(f"Column '{column}' has no missing values.")

    return None


def process_numeric_column(st, df, selected_column, numeric_processor):
    """Process a numeric column with various operations"""
    # For numeric columns
    operation = st.selectbox(
        "Select operation:",
        ["Basic Statistics", "Normalize (0-1)", "Standardize (z-score)",
         "Replace Missing Values", "Binning", "Log Transform", "Handle Outliers"]
    )

    # Create a preview container
    preview_container = st.container()

    if operation == "Basic Statistics":
        stats = numeric_processor.get_basic_statistics(df, selected_column)
        st.write(stats)

    elif operation == "Normalize (0-1)":
        # Get parameters for preview
        min_val = df[selected_column].min()
        max_val = df[selected_column].max()
        range_val = max_val - min_val
        if range_val == 0:
            st.warning("Cannot normalize - all values are the same!")
        else:
            # Get normalized values for preview without adding column
            normalized = numeric_processor.normalize(df, selected_column, add_as_new_column=False, record_to_pipeline=False)
            # Use our utility function instead of the manual preview setup
            create_transform_preview(
                st,
                df,
                selected_column,
                f"{selected_column}_normalized",
                normalized,
                title="Preview of normalized data"
            )
            if st.button("Apply Normalization"):
                st.session_state.data = numeric_processor.normalize(df, selected_column, record_to_pipeline=True)
                st.success(f"Created new column: {selected_column}_normalized")
                st.rerun()

    elif operation == "Standardize (z-score)":
        # Get parameters for preview
        mean = df[selected_column].mean()
        std = df[selected_column].std()
        if std == 0:
            st.warning("Cannot standardize - standard deviation is zero!")
        else:
            # Get standardized values for preview without adding column
            standardized = numeric_processor.standardize(df, selected_column, add_as_new_column=False, record_to_pipeline=False)
            # Use our utility function
            create_transform_preview(
                st,
                df,
                selected_column,
                f"{selected_column}_standardized",
                standardized,
                title="Preview of standardized data"
            )
            if st.button("Apply Standardization"):
                st.session_state.data = numeric_processor.standardize(df, selected_column, record_to_pipeline=True)
                st.success(f"Created new column: {selected_column}_standardized")
                st.rerun()

    elif operation == "Replace Missing Values":
        # Call the improved missing values UI function
        result_df = create_missing_values_ui(st, df, selected_column, numeric_processor, is_numeric=True)
        if result_df is not None:
            st.session_state.data = result_df
            st.rerun()

    elif operation == "Binning":
        bins = st.slider("Number of bins:", min_value=2, max_value=20, value=4)
        # Preview of bin edges
        bin_edges = numeric_processor.get_bin_edges(df, selected_column, bins)
        st.write("Bin edges:")
        st.write(bin_edges)
        # Options for binning method
        bin_method = st.radio("Binning method:", ["Equal width", "Equal frequency"], horizontal=True)
        # Preview binned data
        if bin_method == "Equal width":
            binned_values = pd.cut(df[selected_column], bins)
        else:  # Equal frequency
            binned_values = pd.qcut(df[selected_column], bins, duplicates='drop')
        # Use our utility function
        create_transform_preview(
            st,
            df,
            selected_column,
            f"{selected_column}_binned",
            binned_values,
            title="Preview of binned data"
        )
        if st.button("Apply Binning"):
            st.session_state.data = numeric_processor.apply_binning(
                df, selected_column, bins, bin_method, record_to_pipeline=True)
            st.success(f"Created new column: {selected_column}_binned")
            st.rerun()

    elif operation == "Log Transform":
        # Check for non-positive values
        min_val = df[selected_column].min()
        has_non_positive = min_val <= 0
        if has_non_positive:
            st.warning(f"Column contains non-positive values (min: {min_val}). Will apply offset.")
            offset_method = st.radio("Offset method:", ["Add min + 1", "Custom offset"], horizontal=True)
            if offset_method == "Custom offset":
                offset = st.number_input("Enter offset value:",
                                         value=float(abs(min_val) + 1 if min_val <= 0 else 1))
            else:
                offset = abs(min_val) + 1 if min_val <= 0 else 1
        else:
            offset = 0
        # Create log transformed values
        log_transformed = np.log(df[selected_column] + offset)
        # Use our utility function
        create_transform_preview(
            st,
            df,
            selected_column,
            f"{selected_column}_log",
            log_transformed,
            title="Preview of log transformed data"
        )
        if st.button("Apply Log Transform"):
            st.session_state.data = numeric_processor.apply_log_transform(
                df, selected_column, offset, record_to_pipeline=True)
            if offset > 0:
                st.info(f"Added offset of {offset} before applying log transform.")
            st.success(f"Created new column: {selected_column}_log")
            st.rerun()

    elif operation == "Handle Outliers":
        # Calculate IQR
        outlier_stats = numeric_processor.calculate_outlier_bounds(df, selected_column)
        outliers = numeric_processor.detect_outliers(df, selected_column,
                                                     outlier_stats['lower_bound'], outlier_stats['upper_bound'])
        st.write(f"Potential outliers detected: {len(outliers)} values")
        st.write(f"Lower bound: {outlier_stats['lower_bound']}, Upper bound: {outlier_stats['upper_bound']}")
        method = st.radio("Outlier handling method:",
                          ["Cap at bounds", "Remove outliers", "Replace with NaN"], horizontal=True)
        # Preview outlier handled data
        preview_values = None
        preview_column = None
        if method == "Cap at bounds":
            preview_values = df[selected_column].clip(
                lower=outlier_stats['lower_bound'], upper=outlier_stats['upper_bound'])
            preview_column = f"{selected_column}_capped"
        elif method == "Replace with NaN":
            outlier_mask = (df[selected_column] < outlier_stats['lower_bound']) | (
                    df[selected_column] > outlier_stats['upper_bound'])
            vals = df[selected_column].copy()
            vals[outlier_mask] = np.nan
            preview_values = vals
            preview_column = f"{selected_column}_no_outliers"
        # Use our utility function for methods that don't remove rows
        if method != "Remove outliers":
            create_transform_preview(
                st,
                df,
                selected_column,
                preview_column,
                preview_values,
                title="Preview of data after outlier handling"
            )
        else:
            # For "Remove outliers" we need to show a filtered DataFrame
            mask = (df[selected_column] >= outlier_stats['lower_bound']) & (
                    df[selected_column] <= outlier_stats['upper_bound'])
            st.subheader("Preview of data after outlier handling")
            st.dataframe(df[mask].head(), use_container_width=True)
        if st.button("Handle Outliers"):
            st.session_state.data = numeric_processor.handle_outliers(
                df, selected_column, outlier_stats['lower_bound'],
                outlier_stats['upper_bound'], method , record_to_pipeline=True)
            if method == "Remove outliers":
                st.success(f"Removed {len(outliers)} outlier rows from dataset")
            else:
                st.success(
                    f"Created new column: {selected_column}_{'capped' if method == 'Cap at bounds' else 'no_outliers'}")
            st.rerun()


def process_categorical_column(st, df, selected_column, categorical_processor):
    """Process a categorical column with various operations"""
    # For categorical columns
    operation = st.selectbox(
        "Select operation:",
        ["Frequency Count", "One-Hot Encode", "Label Encode",
         "Replace Missing Values", "Merge Categories"]
    )

    # Create a preview container
    preview_container = st.container()

    if operation == "Frequency Count":
        frequency_stats = categorical_processor.get_frequency_stats(df, selected_column)
        st.subheader(
            f"Most common in {selected_column}: {frequency_stats['most_common_value']} (count: {frequency_stats['most_common']})")
        # Show frequency distribution
        st.bar_chart(frequency_stats['value_counts'])

    elif operation == "One-Hot Encode":
        # Preview how many columns will be created
        unique_values = categorical_processor.get_unique_count(df, selected_column)
        st.write(f"This will create {unique_values} new columns.")
        if unique_values > 20:
            st.warning(f"Column has {unique_values} unique values. One-hot encoding will create many columns.")
        # Option to keep or drop original
        keep_original = st.checkbox("Keep original column", value=False)
        # Preview one-hot encoded data - this is a special case that adds multiple columns
        # so we'll use the standard preview container approach
        with st.container():
            st.subheader("Preview of one-hot encoded data")
            preview_df = df.copy()
            one_hot = pd.get_dummies(preview_df[selected_column], prefix=selected_column)
            for col in one_hot.columns:
                preview_df[col] = one_hot[col]
            if not keep_original:
                preview_df = preview_df.drop(columns=[selected_column])
            st.dataframe(preview_df.head(), use_container_width=True)
        if st.button("Apply One-Hot Encoding"):
            st.session_state.data = categorical_processor.apply_one_hot_encoding(
                df, selected_column, keep_original, record_to_pipeline=True)
            st.success(f"Added {unique_values} one-hot encoded columns for {selected_column}")
            st.rerun()

    elif operation == "Label Encode":
        # Show mapping preview
        mapping = categorical_processor.get_label_mapping(df, selected_column)
        st.write("Label Encoding Mapping:")

        # Get number of unique values
        num_values = len(mapping)
        if num_values <= 10:
            # Show all mappings if 10 or fewer
            for value, code in mapping.items():
                st.write(f"{value} → {code}")
        else:
            # Show first 5 and last 5 if more than 10
            st.write("First 5 mappings:")
            for i, (value, code) in enumerate(mapping.items()):
                if i < 5:
                    st.write(f"{value} → {code}")
                else:
                    break

            st.write(f"... {num_values - 10} more mappings ...")

            st.write("Last 5 mappings:")
            items = list(mapping.items())
            for value, code in items[-5:]:
                st.write(f"{value} → {code}")

        # Preview label encoded data
        encoded_values = df[selected_column].map(mapping)
        # Use our utility function
        create_transform_preview(
            st,
            df,
            selected_column,
            f"{selected_column}_encoded",
            encoded_values,
            title="Preview of label encoded data"
        )
        if st.button("Apply Label Encoding"):
            st.session_state.data = categorical_processor.apply_label_encoding(
                df, selected_column, mapping, record_to_pipeline=True)
            st.success(f"Created new column: {selected_column}_encoded")
            st.rerun()

    elif operation == "Replace Missing Values":
        # Call the improved missing values UI function
        result_df = create_missing_values_ui(st, df, selected_column, categorical_processor, is_numeric=False)
        if result_df is not None:
            st.session_state.data = result_df
            st.rerun()

    elif operation == "Merge Categories":
        unique_values = df[selected_column].unique().tolist()
        # First multiselect for categories to merge
        categories_to_merge = st.multiselect(
            "Select categories to merge:",
            unique_values,
            default=[]
        )
        # Input for new category nam
        if categories_to_merge:
            new_category_name = st.text_input("New category name:", "Merged_Category")
            # Create the merged values
            new_values = df[selected_column].copy()
            mask = new_values.isin(categories_to_merge)
            new_values[mask] = new_category_name
            # Use our utility function
            create_transform_preview(
                st,
                df,
                selected_column,
                f"{selected_column}_merged",
                new_values,
                title="Preview of data with merged categories"
            )
            if st.button("Merge Categories"):
                st.session_state.data = categorical_processor.merge_categories(
                    df, selected_column, categories_to_merge, new_category_name, record_to_pipeline=True)
                st.success(f"Created new column: {selected_column}_merged")
                st.rerun()


def batch_process_missing_values(st, df, selected_columns, column_types, processor):
    """
    Batch process missing values for multiple columns of the same type

    Args:
        st: Streamlit instance
        df: DataFrame
        selected_columns: List of columns to process (all the same type)
        column_types: Dictionary mapping column names to their types
        processor: The processor instance for the column type

    Returns:
        Modified DataFrame if changes are applied, otherwise None
    """
    if not selected_columns:
        st.warning("Please select at least one column.")
        return None

    # Determine if we're working with numeric, categorical, or datetime columns
    column_type = column_types.get(selected_columns[0])
    is_numeric = column_type == 'numeric'
    is_datetime = column_type == 'datetime'
    is_categorical = column_type == 'categorical'

    # Show counts of selected columns
    st.write(f"Selected {len(selected_columns)} {column_type} columns.")

    # Option to replace in original columns or create new ones
    replacement_mode = st.radio(
        "How to replace missing values:",
        ["Replace in original columns", "Create new columns with filled values"],
        index=0
    )
    inplace = replacement_mode == "Replace in original columns"

    # Get replacement strategy based on column type
    if is_numeric:
        strategy = st.selectbox(
            "Replacement strategy:",
            ["Mean", "Median", "Mode", "Zero", "Custom Value"]
        )

        # Custom value if selected
        value = None
        if strategy == "Custom Value":
            value = st.number_input("Custom replacement value:")
        elif strategy == "Zero":
            value = 0

    elif is_datetime:
        strategy = st.selectbox(
            "Replacement strategy:",
            ["Min Date", "Max Date", "Median Date", "Forward Fill", "Backward Fill", "Custom Date"]
        )

        # Map strategy to method
        method_map = {
            "Min Date": "fill_na",
            "Max Date": "fill_na",
            "Median Date": "fill_na",
            "Forward Fill": "forward_fill",
            "Backward Fill": "backward_fill",
            "Custom Date": "fill_na"
        }

        method = method_map[strategy]

        # Custom value if selected
        value = None
        if strategy == "Custom Date":
            value = st.date_input("Custom replacement date:")

    else:  # categorical
        strategy = st.selectbox(
            "Replacement strategy:",
            ["Most Frequent Value", "Custom Value"]
        )

        # Custom value if selected
        value = None
        if strategy == "Custom Value":
            value = st.text_input("Custom replacement value:", "Unknown")

    # Preview section
    with st.expander("Preview of missing values replacement", expanded=True):
        preview_df = df.copy()

        # Create a dataframe to show replacement info
        stats_data = []

        for col in selected_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                # Determine replacement value or method
                if is_numeric:
                    if strategy == "Mean":
                        col_value = df[col].mean()
                    elif strategy == "Median":
                        col_value = df[col].median()
                    elif strategy == "Mode":
                        mode_result = df[col].mode()
                        col_value = mode_result[0] if not mode_result.empty else 0
                    elif strategy == "Zero":
                        col_value = 0
                    else:  # Custom Value
                        col_value = value

                elif is_datetime:
                    if method == "fill_na":
                        if strategy == "Min Date":
                            col_value = processor.get_replacement_value(df, col, 'min')
                        elif strategy == "Max Date":
                            col_value = processor.get_replacement_value(df, col, 'max')
                        elif strategy == "Median Date":
                            col_value = processor.get_replacement_value(df, col, 'median')
                        else:  # Custom Date
                            col_value = processor.get_replacement_value(df, col, 'custom', value)
                    else:
                        # For methods that don't use a specific value
                        col_value = f"Using {strategy}"

                else:  # categorical
                    if strategy == "Most Frequent Value":
                        mode_result = df[col].mode()
                        col_value = mode_result[0] if not mode_result.empty else "Unknown"
                    else:  # Custom Value
                        col_value = value

                # Apply to preview dataframe
                if is_datetime and method != "fill_na":
                    # Handle methods that don't use a specific value
                    if method == "forward_fill":
                        if inplace:
                            preview_df[col] = preview_df[col].ffill()
                        else:
                            preview_df[f"{col}_filled"] = preview_df[col].ffill()
                    elif method == "backward_fill":
                        if inplace:
                            preview_df[col] = preview_df[col].bfill()
                        else:
                            preview_df[f"{col}_filled"] = preview_df[col].bfill()
                else:
                    # Regular value-based replacement
                    if inplace:
                        preview_df[col] = preview_df[col].fillna(col_value)
                    else:
                        preview_df[f"{col}_filled"] = preview_df[col].fillna(col_value)

                # Collect stats for display
                stats_data.append({
                    'Column': col,
                    'Missing Values': missing_count,
                    'Replacement Strategy': strategy,
                    'Replacement Value/Method': str(col_value)
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No missing values in selected columns.")

        # Show sample of filled values
        st.subheader("Sample Rows with Filled Values")

        # Get indices of rows with missing values in selected columns
        missing_mask = df[selected_columns].isna().any(axis=1)
        sample_rows = df[missing_mask].head(10)

        if not sample_rows.empty:
            # Display samples of original vs filled values
            comparison_data = []

            for idx, row in sample_rows.iterrows():
                for col in selected_columns:
                    if pd.isna(row[col]):
                        if inplace:
                            filled_value = preview_df.loc[idx, col]
                        else:
                            filled_value = preview_df.loc[idx, f"{col}_filled"]

                        comparison_data.append({
                            'Row': idx,
                            'Column': col,
                            'Original Value': 'NA',
                            'Filled Value': filled_value
                        })

            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            else:
                st.info("No missing values found in sample rows.")
        else:
            st.info("No rows with missing values found for selected columns.")

    # Apply button
    if st.button("Apply Missing Values Replacement"):
        result_df = df.copy()
        applied_cols = []

        # Process columns
        for col in selected_columns:
            missing_count = processor.get_missing_count(df, col)
            if missing_count > 0:
                try:
                    if is_numeric or is_categorical:
                        # Determine replacement value
                        if is_numeric:
                            if strategy == "Custom Value":
                                col_value = value
                            else:
                                col_value = processor.get_replacement_value(df, col, strategy.lower())
                        else:  # categorical
                            if strategy == "Most Frequent Value":
                                col_value = processor.get_replacement_value(df, col, 'most_frequent')
                            else:  # Custom Value
                                col_value = value

                        # Apply using the processor method
                        result_df = processor.replace_missing_values(
                            result_df, col, col_value, inplace=inplace, record_to_pipeline=True
                        )

                    elif is_datetime:
                        if method == "fill_na":
                            # Get appropriate value based on strategy
                            if strategy == "Min Date":
                                col_value = processor.get_replacement_value(df, col, 'min')
                            elif strategy == "Max Date":
                                col_value = processor.get_replacement_value(df, col, 'max')
                            elif strategy == "Median Date":
                                col_value = processor.get_replacement_value(df, col, 'median')
                            else:  # Custom Date
                                col_value = processor.get_replacement_value(df, col, 'custom', value)

                            # Apply using the processor method
                            result_df = processor.replace_missing_values(
                                result_df, col, col_value, method='fill_na',
                                inplace=inplace, record_to_pipeline=True
                            )
                        else:
                            # Apply using the processor method with appropriate method
                            result_df = processor.replace_missing_values(
                                result_df, col, None, method=method,
                                inplace=inplace, record_to_pipeline=True
                            )

                    applied_cols.append(col)

                except Exception as e:
                    st.error(f"Error processing column {col}: {str(e)}")

        if applied_cols:
            st.success(f"Applied missing values replacement to {len(applied_cols)} columns.")
            return result_df
        else:
            st.warning("No columns with missing values were processed.")

    return None


def process_multiple_columns(st, df, column_types, numeric_processor, categorical_processor, datetime_processor):
    """
    Process multiple columns at once (batch processing)

    Args:
        st: Streamlit instance
        df: DataFrame to process
        column_types: Dictionary mapping column names to their types
        numeric_processor: NumericProcessor instance
        categorical_processor: CategoricalProcessor instance
        datetime_processor: DatetimeProcessor instance
    """
    # Separate columns by type
    numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
    categorical_columns = [col for col, type_ in column_types.items() if type_ == 'categorical']
    datetime_columns = [col for col, type_ in column_types.items() if type_ == 'datetime']

    # Select operation type first
    operation_type = st.selectbox(
        "Select operation type:",
        ["Numeric Operations", "Categorical Operations", "Datetime Operations"]
    )

    if operation_type == "Numeric Operations" and numeric_columns:
        # Existing numeric operations code...
        # Keep this code as is
        operation = st.selectbox(
            "Select numeric operation:",
            ["Normalize (0-1)", "Standardize (z-score)", "Replace Missing Values", "Log Transform"]
        )

        # Select columns to apply the operation
        selected_columns = st.multiselect(
            "Select columns to process:",
            numeric_columns,
            default=numeric_columns[0] if numeric_columns else []
        )

        if operation == "Normalize (0-1)":
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "Normalize",
                numeric_processor,
                numeric_processor.normalize,
                add_as_new_column=True
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Standardize (z-score)":
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "Standardize",
                numeric_processor,
                numeric_processor.standardize,
                add_as_new_column=True
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Replace Missing Values":
            # Use our new batch processing for missing values
            result = batch_process_missing_values(
                st,
                df,
                selected_columns,
                column_types,
                numeric_processor
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Log Transform":
            # Option for handling non-positive values
            offset_strategy = st.radio(
                "Handling non-positive values:",
                ["Add min + 1 to each column", "Use a global offset"],
                horizontal=True
            )

            global_offset = None
            if offset_strategy == "Use a global offset":
                global_offset = st.number_input("Enter global offset value:", value=1.0)

            # Define the transform function for log transform
            def log_transform_function(df, column, record_to_pipeline=False):
                min_val = df[column].min()
                has_non_positive = min_val <= 0

                if offset_strategy == "Add min + 1 to each column" and has_non_positive:
                    offset = abs(min_val) + 1
                elif offset_strategy == "Use a global offset":
                    offset = global_offset
                else:
                    offset = 0

                return numeric_processor.apply_log_transform(df, column, offset, record_to_pipeline=record_to_pipeline)

            # Use batch processing with our custom transform function
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "Log Transform",
                numeric_processor,
                log_transform_function
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

    elif operation_type == "Categorical Operations" and categorical_columns:
        # Existing categorical operations code...
        # Keep this code as is
        operation = st.selectbox(
            "Select categorical operation:",
            ["One-Hot Encode", "Label Encode", "Replace Missing Values"]
        )

        # Select columns to apply the operation
        selected_columns = st.multiselect(
            "Select columns to process:",
            categorical_columns,
            default=categorical_columns[0] if categorical_columns else []
        )

        if operation == "One-Hot Encode":
            # Option to keep or drop original
            keep_original = st.checkbox("Keep original columns", value=False)

            # Define the transform function for one-hot encoding
            def one_hot_encode_function(df, column, record_to_pipeline=False):
                return categorical_processor.apply_one_hot_encoding(
                    df, column, keep_original, record_to_pipeline=record_to_pipeline
                )

            # Use batch processing with our custom transform function
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "One-Hot Encode",
                categorical_processor,
                one_hot_encode_function
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Label Encode":
            # Define the transform function for label encoding
            def label_encode_function(df, column, record_to_pipeline=False):
                mapping = categorical_processor.get_label_mapping(df, column)
                return categorical_processor.apply_label_encoding(
                    df, column, mapping, record_to_pipeline=record_to_pipeline
                )

            # Use batch processing with our custom transform function
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "Label Encode",
                categorical_processor,
                label_encode_function
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Replace Missing Values":
            # Use our new batch processing for missing values
            result = batch_process_missing_values(
                st,
                df,
                selected_columns,
                column_types,
                categorical_processor
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

    elif operation_type == "Datetime Operations" and datetime_columns:
        # New code for datetime batch operations
        operation = st.selectbox(
            "Select datetime operation:",
            ["Extract Components", "Convert to Time Since", "Replace Missing Values"]
        )

        # Select columns to apply the operation
        selected_columns = st.multiselect(
            "Select columns to process:",
            datetime_columns,
            default=datetime_columns[0] if datetime_columns else []
        )

        if operation == "Extract Components":
            # Define components to extract
            components = ["Year", "Month", "Day", "Weekday", "Quarter"]
            st.write("This will extract the following components:", ", ".join(components))

            # Define the transform function for extracting components
            def extract_components_function(df, column, record_to_pipeline=False):
                return datetime_processor.extract_components(
                    df, column, record_to_pipeline=record_to_pipeline
                )

            # Use batch processing
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "Extract Components",
                datetime_processor,
                extract_components_function
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Convert to Time Since":
            # Option for reference date
            reference_date_option = st.radio(
                "Reference date option:",
                ["Use earliest date in each column", "Use custom reference date"],
                horizontal=True
            )

            reference_date = None
            if reference_date_option == "Use custom reference date":
                reference_date = st.date_input("Select reference date")

            # Define the transform function
            def convert_to_time_since_function(df, column, record_to_pipeline=False):
                ref_date = reference_date if reference_date_option == "Use custom reference date" else None
                return datetime_processor.convert_to_time_since(
                    df, column, ref_date, record_to_pipeline=record_to_pipeline
                )

            # Use batch processing
            result = batch_process_columns(
                st,
                df,
                selected_columns,
                "Convert to Time Since",
                datetime_processor,
                convert_to_time_since_function
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

        elif operation == "Replace Missing Values":
            # Use batch processing for missing values
            result = batch_process_missing_values(
                st,
                df,
                selected_columns,
                column_types,
                datetime_processor
            )

            if result is not None:
                st.session_state.data = result
                st.rerun()

    # If no columns of the selected type are available
    elif operation_type == "Numeric Operations" and not numeric_columns:
        st.warning("No numeric columns detected in the dataset.")
    elif operation_type == "Categorical Operations" and not categorical_columns:
        st.warning("No categorical columns detected in the dataset.")
    elif operation_type == "Datetime Operations" and not datetime_columns:
        st.warning("No datetime columns detected in the dataset.")


def process_datetime_column(st, df, selected_column, datetime_processor):
    # For datetime columns
    operation = st.selectbox(
        "Select operation:",
        ["Extract Components", "Convert to Time Since"]
    )

    # Create a preview container
    preview_container = st.container()

    if operation == "Extract Components":
        # Show which components will be extracted
        components = ["Year", "Month", "Day", "Weekday", "Quarter"]
        st.write("This will extract the following components:", ", ".join(components))

        # Preview extracted components
        with preview_container:
            st.subheader("Preview of extracted datetime components")

            # Convert to datetime if needed
            preview_df = df.copy()
            if not pd.api.types.is_datetime64_dtype(preview_df[selected_column]):
                preview_df[selected_column] = pd.to_datetime(preview_df[selected_column], errors='coerce')

            # Generate preview
            preview_df[f"{selected_column}_year"] = preview_df[selected_column].dt.year
            preview_df[f"{selected_column}_month"] = preview_df[selected_column].dt.month
            preview_df[f"{selected_column}_day"] = preview_df[selected_column].dt.day
            preview_df[f"{selected_column}_weekday"] = preview_df[selected_column].dt.weekday
            preview_df[f"{selected_column}_quarter"] = preview_df[selected_column].dt.quarter

            st.dataframe(preview_df.head(), use_container_width=True)

        if st.button("Extract Components"):
            st.session_state.data = datetime_processor.extract_components(df, selected_column, record_to_pipeline=True)
            st.success(f"Created new component columns for {selected_column}")
            st.rerun()

    elif operation == "Convert to Time Since":
        # Option to set reference date
        use_custom_reference = st.checkbox("Use custom reference date", value=False)
        reference_date = None

        if use_custom_reference:
            reference_date = st.date_input("Select reference date")
        else:
            # Convert to datetime if needed for displaying min date
            if not pd.api.types.is_datetime64_dtype(df[selected_column]):
                temp_col = pd.to_datetime(df[selected_column], errors='coerce')
                min_date = temp_col.min()
            else:
                min_date = df[selected_column].min()

            st.write(f"Will use earliest date as reference: {min_date.date()}")

        # Preview time since conversion
        with preview_container:
            st.subheader("Preview of time since conversion")

            # Convert to datetime if needed
            preview_df = df.copy()
            if not pd.api.types.is_datetime64_dtype(preview_df[selected_column]):
                preview_df[selected_column] = pd.to_datetime(preview_df[selected_column], errors='coerce')

            # Calculate reference date
            if use_custom_reference:
                ref_date = pd.to_datetime(reference_date)
            else:
                ref_date = preview_df[selected_column].min()

            # Generate preview
            preview_df[f"{selected_column}_days_since"] = (preview_df[
                                                               selected_column] - ref_date).dt.total_seconds() / (
                                                                      24 * 60 * 60)

            st.dataframe(preview_df.head(), use_container_width=True)

        if st.button("Convert to Days Since"):
            st.session_state.data = datetime_processor.convert_to_time_since(df, selected_column, reference_date, record_to_pipeline=True)
            st.success(f"Created new column: {selected_column}_days_since")
            st.rerun()




# Configure page
# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Data Preprocessing"
)

# Apply the CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)

# Additional CSS for better styling
st.markdown("""
<style>
    .operation-selector {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .operation-card {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border-left: 3px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .column-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 1rem;
    }

    .stat-item {
        background-color: rgba(59, 130, 246, 0.1);
        border-radius: 6px;
        padding: 8px 12px;
        flex: 1;
        min-width: 150px;
    }

    .stat-label {
        color: #9CA3AF;
        font-size: 0.8rem;
    }

    .stat-value {
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .action-button {
        margin-top: 10px;
    }

    .tab-subheader {
        font-size: 1.1rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
        padding-bottom: 0.3rem;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(to right, rgba(59, 130, 246, 0), rgba(59, 130, 246, 0.3), rgba(59, 130, 246, 0));
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Page header with back navigation
st.markdown('<div class="page-header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
    if st.button('← ', key='back_button', help="Return to data quality page"):
        st.switch_page("pages/2_data_quality.py")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="margin: 0; color: white;">Data Preprocessing</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Check if data exists in session state
if 'data' not in st.session_state:
    st.session_state.data = None

if st.session_state.data is None:
    st.markdown('<div class="card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
    st.warning("No data loaded. Please upload a dataset first.")
    if st.button('Go to Data Upload', use_container_width=True):
        st.switch_page("pages/1_upload_data.py")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Create tabs for different functionalities
    preprocess_tab, feature_tab, pipeline_tab = st.tabs(["Preprocessing", "Feature Engineering", "Pipeline Manager"])

    with preprocess_tab:
        # Dataset Overview Card
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Header with download button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
        with col2:
            # Add download button for the current dataframe
            csv = st.session_state.data.to_csv(index=False).encode('utf-8')
            original_filename = st.session_state.file_name.rsplit('.', 1)[0] if hasattr(st.session_state,
                                                                                        'file_name') else "data"
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{original_filename}_processed.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<span class="metric-label">Rows</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.data.shape[0]:,}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<span class="metric-label">Columns</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.data.shape[1]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            missing_count = st.session_state.data.isna().sum().sum()
            missing_pct = round(missing_count / (st.session_state.data.shape[0] * st.session_state.data.shape[1]) * 100,
                                2)
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<span class="metric-label">Missing Values</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{missing_count:,}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<span class="metric-label">Missing Percentage</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{missing_pct}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Dataset preview
        with st.expander("View Dataset Preview", expanded=True):
            # Option to show all columns or select specific ones
            all_columns = st.checkbox("Show all columns", value=True)
            if all_columns:
                display_columns = st.session_state.data.columns
            else:
                display_columns = st.multiselect("Select columns to display", st.session_state.data.columns,
                                                 default=list(st.session_state.data.columns[:5]))

            # Display configurable dataframe
            st.dataframe(st.session_state.data[display_columns], use_container_width=True)
            st.markdown(
                "<div style='text-align: right; color: #9CA3AF; font-size: 0.8rem;'>Showing first 10 rows</div>",
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Missing Values Analysis Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Missing Values Analysis</div>', unsafe_allow_html=True)

        # Calculate missing values per column
        missing_values = st.session_state.data.isna().sum()
        missing_columns = missing_values[missing_values > 0]

        if len(missing_columns) == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            # Create a DataFrame for display
            missing_df = pd.DataFrame({
                'Column': missing_columns.index,
                'Missing Values': missing_columns.values,
                'Percentage': (missing_columns.values / len(st.session_state.data) * 100).round(2)
            }).sort_values('Missing Values', ascending=False)

            # Display a bar chart of missing values
            with st.expander("View Missing Values Chart", expanded=False):
                missing_columns_sorted = missing_df.sort_values('Missing Values')
                st.bar_chart(missing_columns_sorted.set_index('Column')['Percentage'], use_container_width=True)

            # Display missing values table
            st.dataframe(missing_df, use_container_width=True)

            # Manage Rows & Columns checkbox
            if st.checkbox("Manage Rows & Columns", value=False):
                modified_df = manage_rows_columns(st.session_state.data, missing_columns)
                if modified_df is not None:
                    st.session_state.data = modified_df
                    st.rerun()  # Rerun the app to reflect the changes

        st.markdown('</div>', unsafe_allow_html=True)

        # Column Operations Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Column Operations</div>', unsafe_allow_html=True)

        # Initialize processors
        data_processor = DataProcessor()
        numeric_processor = NumericProcessor()
        categorical_processor = CategoricalProcessor()
        datetime_processor = DatetimeProcessor()

        # Detect column types
        column_types = detect_column_types(st.session_state.data)

        # Column type summary
        numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
        categorical_columns = [col for col, type_ in column_types.items() if type_ == 'categorical']
        datetime_columns = [col for col, type_ in column_types.items() if type_ == 'datetime']

        st.markdown('<div class="column-stats">', unsafe_allow_html=True)
        st.markdown(f'''
            <div class="stat-item">
                <div class="stat-label">Numeric Columns</div>
                <div class="stat-value">{len(numeric_columns)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Categorical Columns</div>
                <div class="stat-value">{len(categorical_columns)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Datetime Columns</div>
                <div class="stat-value">{len(datetime_columns)}</div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="operation-selector">', unsafe_allow_html=True)
        # Radio button to choose between single column or batch processing
        processing_mode = st.radio(
            "Processing mode:",
            ["Single column", "Multiple columns (batch processing)"],
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if processing_mode == "Single column":
            # Get columns from the dataframe
            columns = st.session_state.data.columns.tolist()

            # Let user select a column
            selected_column = st.selectbox("Select a column to operate on:", columns)

            # Check the data type to offer appropriate functions
            column_type = 'categorical'  # Default

            # Try to detect column type
            if pd.api.types.is_numeric_dtype(st.session_state.data[selected_column]):
                column_type = 'numeric'
            elif pd.api.types.is_datetime64_dtype(st.session_state.data[selected_column]):
                column_type = 'datetime'
            else:
                # Try to detect if it can be converted to datetime
                try:
                    pd.to_datetime(st.session_state.data[selected_column], errors='raise')
                    column_type = 'datetime'
                except Exception:
                    column_type = 'categorical'

            # Column info box
            st.markdown(f'''
            <div class="operation-card">
                <h3 style="margin-top: 0;">{selected_column}</h3>
                <div style="display: flex; gap: 20px;">
                    <div>
                        <div class="stat-label">Type</div>
                        <div class="stat-value">{column_type.capitalize()}</div>
                    </div>
                    <div>
                        <div class="stat-label">Unique Values</div>
                        <div class="stat-value">{st.session_state.data[selected_column].nunique()}</div>
                    </div>
                    <div>
                        <div class="stat-label">Missing Values</div>
                        <div class="stat-value">{st.session_state.data[selected_column].isna().sum()}</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Process based on detected type
            if column_type == 'numeric':
                process_numeric_column(st, st.session_state.data, selected_column, numeric_processor)
            elif column_type == 'datetime':
                process_datetime_column(st, st.session_state.data, selected_column, datetime_processor)
            else:
                process_categorical_column(st, st.session_state.data, selected_column, categorical_processor)
        else:
            # Batch processing for multiple columns
            process_multiple_columns(st, st.session_state.data, column_types, numeric_processor, categorical_processor,
                                     datetime_processor)
        st.markdown('</div>', unsafe_allow_html=True)

    with feature_tab:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Engineering</div>', unsafe_allow_html=True)

        # Feature engineering explanation
        with st.expander("About Feature Engineering", expanded=False):
            st.markdown("""
            Feature engineering is the process of creating new features from existing data to improve model performance. 
            Select a technique below to create new features derived from your dataset.

            **Common techniques include:**
            - Creating polynomial features
            - Extracting components from dates
            - Creating interaction terms between features
            - Binning continuous variables
            - Text feature extraction
            """)

        # Call the feature engineering UI
        column_types = detect_column_types(st.session_state.data)
        fe_result = feature_engineering_ui(st, st.session_state.data, column_types)
        if fe_result is not None:
            st.session_state.data = fe_result
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with pipeline_tab:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pipeline Manager</div>', unsafe_allow_html=True)

        # Pipeline manager explanation
        with st.expander("About Preprocessing Pipelines", expanded=False):
            st.markdown("""
            Create, save, and apply sequences of preprocessing steps that can be reused on similar datasets. 
            Pipelines help ensure consistency and reproducibility in your data preprocessing workflow.

            **Benefits of using pipelines:**
            - Reproducibility: Apply the same steps to different datasets
            - Efficiency: Run multiple operations with a single click
            - Documentation: Keep track of all transformations applied to your data
            - Collaboration: Share pipelines with team members
            - Version control: Save different preprocessing strategies
            """)

        processors = {
            'data': DataProcessor(),
            'numeric': NumericProcessor(),
            'categorical': CategoricalProcessor(),
            'datetime': DatetimeProcessor(),
            'feature': FeatureEngineeringProcessor()
        }

        # Call the pipeline manager UI
        pipeline_result = pipeline_manager_ui(st, st.session_state.data, processors)
        if pipeline_result is not None:
            st.session_state.data = pipeline_result
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Navigation buttons at the bottom
    st.markdown('<div style="display: flex; justify-content: space-between; margin-top: 2rem;">',
                unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('← Back to Data Quality', key='back_to_quality', use_container_width=True):
            st.switch_page("pages/2_data_quality.py")
    with col2:
        if st.button('Continue to Visualization →', key='continue_btn', use_container_width=True):
            st.switch_page("pages/4_visualization.py")
    st.markdown('</div>', unsafe_allow_html=True)