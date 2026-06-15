import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_export import export_chart


def display_numeric_summary(viz_data, numeric_columns):
    """
    Display summary statistics for numeric columns

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
    """
    # Select columns for statistics
    selected_columns = st.multiselect(
        "Select columns for statistics:",
        numeric_columns,
        default=numeric_columns,
        key="num_summary_cols"
    )

    if not selected_columns:
        st.warning("Please select at least one column.")
        return None, None

    # Calculate descriptive statistics
    stats_df = viz_data[selected_columns].describe().T

    # Add additional statistics
    stats_df['skew'] = viz_data[selected_columns].skew()
    stats_df['kurtosis'] = viz_data[selected_columns].kurtosis()
    stats_df['missing'] = viz_data[selected_columns].isna().sum()
    stats_df['missing_pct'] = (viz_data[selected_columns].isna().sum() / len(viz_data) * 100).round(2)

    # Round values for display
    stats_df = stats_df.round(2)

    # Display statistics table
    st.dataframe(stats_df, use_container_width=True)

    # Create bar chart of means
    fig_mean = px.bar(
        x=stats_df.index,
        y=stats_df['mean'],
        error_y=stats_df['std'],
        title="Mean Values with Standard Deviation",
        labels={'x': 'Variable', 'y': 'Mean Value'}
    )

    st.plotly_chart(fig_mean, use_container_width=True)

    # Show skewness and kurtosis
    col1, col2 = st.columns(2)

    with col1:
        fig_skew = px.bar(
            x=stats_df.index,
            y=stats_df['skew'],
            title="Skewness",
            labels={'x': 'Variable', 'y': 'Skewness'}
        )
        fig_skew.add_hline(y=0, line_dash="dash", line_color="red")
        fig_skew.update_layout(height=400)
        st.plotly_chart(fig_skew, use_container_width=True)

    with col2:
        fig_kurt = px.bar(
            x=stats_df.index,
            y=stats_df['kurtosis'],
            title="Kurtosis",
            labels={'x': 'Variable', 'y': 'Kurtosis'}
        )
        fig_kurt.add_hline(y=3, line_dash="dash", line_color="red")
        fig_kurt.update_layout(height=400)
        st.plotly_chart(fig_kurt, use_container_width=True)

    return fig_mean, "summary_statistics"


def display_categorical_summary(viz_data, categorical_columns):
    """
    Display summary statistics for categorical columns

    Args:
        viz_data: DataFrame containing the data
        categorical_columns: List of categorical column names
    """
    # Select columns for statistics
    selected_columns = st.multiselect(
        "Select columns for summary:",
        categorical_columns,
        default=categorical_columns[:min(len(categorical_columns), 5)],
        key="cat_summary_cols"
    )

    if not selected_columns:
        st.warning("Please select at least one column.")
        return None, None

    # Table to store statistics
    cat_stats = []

    for col in selected_columns:
        # Calculate statistics
        n_unique = viz_data[col].nunique()
        n_missing = viz_data[col].isna().sum()
        missing_pct = (n_missing / len(viz_data) * 100).round(2)

        # Handle empty columns
        if n_unique > 0:
            most_common = viz_data[col].value_counts().index[0] if not viz_data[
                col].value_counts().empty else None
            most_common_count = viz_data[col].value_counts().iloc[0] if not viz_data[
                col].value_counts().empty else 0
        else:
            most_common = None
            most_common_count = 0

        most_common_pct = (most_common_count / len(viz_data) * 100).round(2)

        # Add to stats table
        cat_stats.append({
            'Column': col,
            'Unique Values': n_unique,
            'Missing Values': n_missing,
            'Missing (%)': missing_pct,
            'Most Common': str(most_common),
            'Most Common Count': most_common_count,
            'Most Common (%)': most_common_pct
        })

    # Convert to DataFrame
    stats_df = pd.DataFrame(cat_stats)

    # Display statistics table
    st.dataframe(stats_df, use_container_width=True)

    # Select one column for detailed visualization
    selected_col = st.selectbox("Select column for detailed view:", selected_columns)

    # Calculate value counts for the selected column
    value_counts = viz_data[selected_col].value_counts().reset_index()
    value_counts.columns = [selected_col, 'Count']
    value_counts['Percentage'] = (value_counts['Count'] / len(viz_data) * 100).round(2)

    # Display the counts
    if len(value_counts) > 10:
        st.write(f"Top 10 values (out of {len(value_counts)} unique values):")
        st.dataframe(value_counts.head(10), use_container_width=True)
    else:
        st.dataframe(value_counts, use_container_width=True)

    # Create bar chart of counts if not too many categories
    if len(value_counts) <= 20:
        fig = px.bar(
            value_counts.head(20),
            x=selected_col,
            y='Count',
            text='Percentage',
            title=f"Distribution of {selected_col}"
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        return fig, f"cat_summary_{selected_col}"
    else:
        st.info(f"Bar chart not shown as there are too many ({len(value_counts)}) unique values.")
        return None, None


def display_group_statistics(viz_data, numeric_columns, categorical_columns):
    """
    Display group statistics for numeric variables grouped by categorical variables

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
    """
    if len(categorical_columns) == 0 or len(numeric_columns) == 0:
        st.warning("Need at least one categorical and one numeric column for group statistics.")
        return None, None

    # Select grouping variable
    group_col = st.selectbox("Select grouping variable:", categorical_columns, key="group_stats_cat")

    # Select values to analyze
    selected_columns = st.multiselect(
        "Select numeric columns to analyze by group:",
        numeric_columns,
        default=numeric_columns[:min(len(numeric_columns), 3)],
        key="group_stats_num_cols"
    )

    if not selected_columns:
        st.warning("Please select at least one numeric column.")
        return None, None

    # Select aggregation method
    agg_method = st.multiselect(
        "Select aggregation methods:",
        ["Mean", "Median", "Min", "Max", "Count", "Sum", "Std"],
        default=["Mean", "Median", "Std"],
        key="group_stats_agg"
    )

    agg_dict = {
        "Mean": "mean",
        "Median": "median",
        "Min": "min",
        "Max": "max",
        "Count": "count",
        "Sum": "sum",
        "Std": "std"
    }

    # Convert selected methods to function names
    agg_func = {col: [agg_dict[method] for method in agg_method] for col in selected_columns}

    # Limit categories if too many
    if viz_data[group_col].nunique() > 15:
        st.warning(f"Too many categories in {group_col}. Showing only top 15 by frequency.")
        top_cats = viz_data[group_col].value_counts().nlargest(15).index
        filtered_data = viz_data[viz_data[group_col].isin(top_cats)]
    else:
        filtered_data = viz_data

    try:
        # Group by and aggregate
        grouped_stats = filtered_data.groupby(group_col)[selected_columns].agg(agg_func)

        # Format column names
        grouped_stats.columns = [f"{col}_{agg}" for col, agg in grouped_stats.columns]

        # Display results
        st.dataframe(grouped_stats.round(2), use_container_width=True)

        # Create visualization for the first column and aggregation method
        col = selected_columns[0]

        # Choose the aggregation to plot
        if "Mean" in agg_method:
            agg_to_plot = "mean"
            agg_name = "Mean"
        else:
            agg_to_plot = agg_func[col][0]
            agg_name = agg_method[0]

        # Extract data for plotting
        plot_data = grouped_stats[[f"{col}_{agg_to_plot}"]].reset_index()
        plot_data.columns = [group_col, f"{agg_name} of {col}"]

        # Create bar chart
        fig = px.bar(
            plot_data,
            x=group_col,
            y=f"{agg_name} of {col}",
            title=f"{agg_name} of {col} by {group_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

        return fig, f"group_stats_{group_col}_{col}_{agg_to_plot}"
    except Exception as e:
        st.error(f"Error creating group statistics: {str(e)}")
        return None, None


def display_missing_value_analysis(data):
    """
    Display analysis of missing values

    Args:
        data: DataFrame containing the data
    """
    # Calculate missing values
    missing_df = pd.DataFrame({
        'Column': data.columns,
        'Missing Values': data.isna().sum().values,
        'Percentage': (data.isna().sum().values / len(data) * 100).round(2)
    })

    # Sort by percentage of missing values
    missing_df = missing_df.sort_values('Percentage', ascending=False)

    # Display missing values table
    st.dataframe(missing_df, use_container_width=True)

    # Create bar chart of missing values
    fig = px.bar(
        missing_df,
        x='Column',
        y='Percentage',
        title="Percentage of Missing Values by Column",
        text='Percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Missing values correlation
    if len(missing_df[missing_df['Missing Values'] > 0]) >= 2:
        st.subheader("Missing Value Patterns")

        # Get columns with missing values
        missing_cols = missing_df[missing_df['Missing Values'] > 0]['Column'].tolist()

        if len(missing_cols) > 1:
            try:
                # Create binary mask of missing values
                missing_mask = data[missing_cols].isna().astype(int)

                # Calculate correlation between missing value patterns
                missing_corr = missing_mask.corr()

                # Create heatmap
                fig_corr = px.imshow(
                    missing_corr,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title="Correlation Between Missing Values"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

                st.info("Values close to 1 indicate that missing values in these columns tend to occur together.")

                # Return the correlation heatmap
                return fig_corr, "missing_values_correlation"
            except Exception as e:
                st.error(f"Error creating missing value correlation: {str(e)}")

    return fig, "missing_values_chart"