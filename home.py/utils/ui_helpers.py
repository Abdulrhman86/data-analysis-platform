import streamlit as st
import pandas as pd


def create_transform_preview(st, df, original_column, transformed_column, transformed_values, title=None):
    """
    Creates a preview of transformed data

    Args:
        st: Streamlit instance
        df: Original DataFrame
        original_column: Name of the original column
        transformed_column: Name to use for the transformed column
        transformed_values: Series containing the transformed values
        title: Optional title for the preview

    Returns:
        Preview DataFrame
    """
    preview_container = st.container()
    with preview_container:
        preview_title = title if title else f"Preview of transformed data"
        st.subheader(preview_title)

        preview_df = df.copy()
        preview_df[transformed_column] = transformed_values

        st.dataframe(preview_df.head(10), use_container_width=True)

    return preview_df


def batch_process_columns(st, df, columns, operation_name, processor, transform_func, **kwargs):
    """
    Generic function for batch processing multiple columns

    Args:
        st: Streamlit instance
        df: DataFrame
        columns: List of columns to process
        operation_name: Name of the operation (for UI)
        processor: Processor instance
        transform_func: Function that transforms a column
        **kwargs: Additional parameters for the transform function

    Returns:
        Modified DataFrame if applied, otherwise None
    """
    if not columns:
        st.warning("Please select at least one column.")
        return None

    # Preview container
    with st.container():
        st.subheader(f"Preview of {operation_name}")
        preview_df = df.copy()

        # Apply preview to each column
        success_cols = []
        for col in columns:
            try:
                # Generate preview - Add record_to_pipeline=False for preview
                if 'add_as_new_column' in kwargs:
                    # For functions that support this parameter
                    result = transform_func(df, col, add_as_new_column=False,
                                            record_to_pipeline=False,
                                            **{k: v for k, v in kwargs.items() if k != 'add_as_new_column'})
                    if result is not None:
                        preview_df[f"{col}_{operation_name.lower().replace(' ', '_')}"] = result
                        success_cols.append(col)
                else:
                    # For other transform functions
                    result = transform_func(df, col, record_to_pipeline=False, **kwargs)
                    if result is not None and isinstance(result, pd.DataFrame):
                        # If function returns a DataFrame with new columns
                        new_cols = [c for c in result.columns if c not in df.columns]
                        for new_col in new_cols:
                            preview_df[new_col] = result[new_col]
                        success_cols.append(col)
            except Exception as e:
                st.warning(f"Could not process column {col}: {str(e)}")

        st.dataframe(preview_df.head(10), use_container_width=True)

    # Apply button
    if st.button(f"Apply {operation_name} to Selected Columns"):
        modified_df = df.copy()
        applied_cols = []

        for col in success_cols:
            try:
                # Apply transformation - Add record_to_pipeline=True for actual transformation
                result = transform_func(modified_df, col, record_to_pipeline=True, **kwargs)
                if result is not None:
                    # Update the DataFrame with the result
                    if isinstance(result, pd.DataFrame):
                        # If function returns a DataFrame
                        modified_df = result
                    applied_cols.append(col)
            except Exception as e:
                st.error(f"Error processing column {col}: {str(e)}")

        if applied_cols:
            st.success(f"Applied {operation_name} to: {', '.join(applied_cols)}")
            return modified_df
        else:
            st.error("No columns were processed successfully.")

    return None

