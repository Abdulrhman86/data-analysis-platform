import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.chart_export import export_chart, export_matplotlib_chart


def display_numeric_distributions(viz_data, numeric_columns):
    """
    Display comparative distributions for multiple numeric variables

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
    """
    # Select variables to compare
    selected_columns = st.multiselect(
        "Select numeric columns to compare:",
        numeric_columns,
        default=numeric_columns[:min(len(numeric_columns), 3)],
        key="dist_columns"
    )

    if len(selected_columns) < 1:
        st.warning("Please select at least one column.")
        return None, None

    distribution_type = st.radio(
        "Distribution Type",
        ["Histogram", "KDE", "ECDF (Empirical Cumulative Distribution)"],
        horizontal=True,
        key="comp_dist_type"
    )

    if distribution_type == "Histogram":
        # Create histogram for each selected column
        fig = go.Figure()

        for col in selected_columns:
            fig.add_trace(go.Histogram(
                x=viz_data[col].dropna(),  # Drop NaN for cleaner visualization
                name=col,
                opacity=0.7,
                nbinsx=30
            ))

        # Customize layout
        fig.update_layout(
            title="Comparative Histograms",
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
        return fig, "comparative_histograms"

    elif distribution_type == "KDE":
        try:
            plt.figure(figsize=(10, 6))

            for col in selected_columns:
                sns.kdeplot(viz_data[col].dropna(), label=col, fill=True, alpha=0.3)

            plt.title("Density Plot Comparison")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            export_matplotlib_chart(plt, "kde_comparison")

            # Return None since we're using matplotlib and not returning a plotly figure
            return None, "kde_comparison"
        finally:
            plt.close()  # Ensure figure is closed

    elif distribution_type.startswith("ECDF"):
        # Create ECDF for each selected column
        fig = go.Figure()

        for col in selected_columns:
            # Sort values for ECDF (drop NaN values)
            sorted_values = np.sort(viz_data[col].dropna())
            if len(sorted_values) == 0:
                st.warning(f"Column {col} has no non-NA values.")
                continue

            # Calculate ECDF
            y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

            fig.add_trace(go.Scatter(
                x=sorted_values,
                y=y,
                mode='lines',
                name=col
            ))

        # Customize layout
        fig.update_layout(
            title="Empirical Cumulative Distribution Function",
            xaxis_title="Value",
            yaxis_title="Cumulative Probability",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
        return fig, "ecdf_plot"


def display_category_distribution(viz_data, categorical_columns):
    """
    Display distribution of categories across groups

    Args:
        viz_data: DataFrame containing the data
        categorical_columns: List of categorical column names
    """
    if len(categorical_columns) < 2:
        st.warning("Need at least 2 categorical columns for this analysis.")
        return None, None

    col1, col2 = st.columns(2)
    with col1:
        main_cat = st.selectbox("Select main categorical variable:", categorical_columns, key="cat_dist_main")
    with col2:
        group_cat = st.selectbox(
            "Select grouping variable:",
            [col for col in categorical_columns if col != main_cat],
            key="cat_dist_group"
        )

    # Limit categories if too many
    if viz_data[main_cat].nunique() > 15 or viz_data[group_cat].nunique() > 10:
        st.warning("Too many categories. Showing top categories by frequency.")
        main_cat_top = min(15, viz_data[main_cat].nunique())
        group_cat_top = min(10, viz_data[group_cat].nunique())

        top_main = viz_data[main_cat].value_counts().nlargest(main_cat_top).index
        top_group = viz_data[group_cat].value_counts().nlargest(group_cat_top).index

        filtered_data = viz_data[viz_data[main_cat].isin(top_main) & viz_data[group_cat].isin(top_group)]
    else:
        filtered_data = viz_data

    # Calculate the distribution
    try:
        crosstab = pd.crosstab(
            filtered_data[main_cat],
            filtered_data[group_cat],
            normalize='columns'
        ) * 100

        # Display as a heatmap
        fig = px.imshow(
            crosstab,
            text_auto='.1f',
            labels=dict(x=group_cat, y=main_cat, color="Percentage (%)"),
            title=f"Distribution of {main_cat} across {group_cat} Groups"
        )
        fig.update_layout(
            height=max(400, min(800, len(crosstab) * 30)),
            xaxis_title=group_cat,
            yaxis_title=main_cat
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display the raw counts
        st.subheader("Raw Count Data")
        count_crosstab = pd.crosstab(filtered_data[main_cat], filtered_data[group_cat])
        st.dataframe(count_crosstab, use_container_width=True)

        return fig, f"category_dist_{main_cat}_{group_cat}"
    except Exception as e:
        st.error(f"Error creating distribution: {str(e)}")
        return None, None


def display_distribution_grid(viz_data, numeric_columns, categorical_columns):
    """
    Display a grid of distribution plots

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
    """
    if len(numeric_columns) < 1 or len(categorical_columns) < 1:
        st.warning("Need at least 1 numeric column and 1 categorical column for distribution grid.")
        return None, None

    # Select variables
    numeric_col = st.selectbox("Select numeric variable:", numeric_columns, key="dist_grid_num")
    categorical_col = st.selectbox("Select categorical variable for grouping:", categorical_columns,
                                   key="dist_grid_cat")

    # Get unique categories
    categories = viz_data[categorical_col].unique()

    # Create a grid of histograms
    if len(categories) > 16:
        st.warning(f"Too many categories ({len(categories)}). Showing only the top 16 by frequency.")
        categories = viz_data[categorical_col].value_counts().nlargest(16).index

    # Calculate number of rows/cols for subplots
    n_categories = len(categories)
    n_cols = min(4, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols  # Ceiling division

    try:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[str(cat) for cat in categories]
        )

        for i, category in enumerate(categories):
            row = i // n_cols + 1
            col = i % n_cols + 1

            category_data = viz_data[viz_data[categorical_col] == category][numeric_col].dropna()

            fig.add_trace(
                go.Histogram(
                    x=category_data,
                    name=str(category),
                    nbinsx=20,
                    marker_color='blue',
                    opacity=0.7
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=f"Distribution of {numeric_col} by {categorical_col}",
            showlegend=False,
            height=n_rows * 250
        )

        st.plotly_chart(fig, use_container_width=True)
        return fig, f"dist_grid_{numeric_col}_{categorical_col}"
    except Exception as e:
        st.error(f"Error creating distribution grid: {str(e)}")
        return None, None