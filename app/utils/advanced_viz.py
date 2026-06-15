import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_export import export_chart, export_advanced_chart


def display_parallel_coordinates(viz_data, numeric_columns, categorical_columns=None):
    """
    Display a parallel coordinates plot

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names (optional)
    """
    st.write("Parallel Coordinates Plot shows multiple variables for each data point")

    # Select columns for parallel plot
    selected_columns = st.multiselect(
        "Select columns (3-10 recommended):",
        numeric_columns,
        default=numeric_columns[:min(len(numeric_columns), 5)],
        key="parallel_cols"
    )

    if len(selected_columns) < 3:
        st.warning("Please select at least 3 columns for the parallel coordinates plot.")
        return None, None

    # Option to color by category
    color_by = None
    if categorical_columns:
        use_color = st.checkbox("Color by category", key="parallel_use_color")
        if use_color:
            color_by = st.selectbox("Select category for coloring:", categorical_columns, key="parallel_color")

    # Further sample for performance if needed
    pc_data = viz_data
    if len(pc_data) > 1000:
        pc_data = pc_data.sample(1000, random_state=42)
        st.info("Using a sample of 1000 rows for better performance.")

    try:
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            pc_data,
            dimensions=selected_columns,
            color=color_by,
            title="Parallel Coordinates Plot"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        return fig, "parallel_coordinates"
    except Exception as e:
        st.error(f"Error creating parallel coordinates plot: {str(e)}")
        return None, None


def display_radar_chart(viz_data, numeric_columns, categorical_columns):
    """
    Display a radar (spider) chart

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
    """
    st.write("Radar Chart (Spider Chart) compares multiple variables for different categories")

    if len(categorical_columns) == 0 or len(numeric_columns) < 3:
        st.warning("Radar chart requires at least one categorical column and three numeric columns.")
        return None, None

    # Select category for radar chart
    category_col = st.selectbox(
        "Select category column:",
        categorical_columns,
        key="radar_cat"
    )

    # Select numeric variables
    selected_columns = st.multiselect(
        "Select numeric variables (3-10 recommended):",
        numeric_columns,
        default=numeric_columns[:min(len(numeric_columns), 5)],
        key="radar_num_cols"
    )

    if len(selected_columns) < 3:
        st.warning("Please select at least 3 numeric columns for the radar chart.")
        return None, None

    # Select categories to display
    categories = viz_data[category_col].unique().tolist()

    # Limit number of categories for readability
    if len(categories) > 10:
        st.warning(f"Too many categories ({len(categories)}). Please select specific ones.")
        selected_categories = st.multiselect(
            "Select categories to display:",
            categories,
            default=categories[:5],
            key="radar_selected_cats"
        )
    else:
        selected_categories = st.multiselect(
            "Select categories to display:",
            categories,
            default=categories,
            key="radar_selected_cats"
        )

    if not selected_categories:
        st.warning("Please select at least one category to display.")
        return None, None

    try:
        # Calculate mean values for each category and variable
        radar_data = viz_data.groupby(category_col)[selected_columns].mean()

        # Normalize data for radar chart (0-1 scale)
        radar_data_norm = radar_data.copy()
        for col in selected_columns:
            min_val = radar_data[col].min()
            max_val = radar_data[col].max()
            if max_val > min_val:  # Avoid division by zero
                radar_data_norm[col] = (radar_data[col] - min_val) / (max_val - min_val)

        # Filter for selected categories
        radar_data_norm = radar_data_norm.loc[selected_categories]

        # Create radar chart
        fig = go.Figure()

        for category in selected_categories:
            fig.add_trace(go.Scatterpolar(
                r=radar_data_norm.loc[category].values.tolist() + [
                    radar_data_norm.loc[category].values[0]],
                theta=selected_columns + [selected_columns[0]],
                fill='toself',
                name=str(category)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Radar Chart (Normalized Values)",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display the original mean values
        st.subheader("Mean Values (Before Normalization)")
        st.dataframe(radar_data.loc[selected_categories].round(2), use_container_width=True)

        return fig, f"radar_chart_{category_col}"
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None, None


def display_3d_scatter(viz_data, numeric_columns, categorical_columns=None):
    """
    Display a 3D scatter plot

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names (optional)
    """
    st.write("3D Scatter Plot visualizes relationships between three numeric variables")

    if len(numeric_columns) < 3:
        st.warning("Need at least 3 numeric columns for a 3D scatter plot.")
        return None, None

    col1, col2, col3 = st.columns(3)

    with col1:
        x_col = st.selectbox("X-axis:", numeric_columns, index=0, key="3d_x")
    with col2:
        y_col = st.selectbox("Y-axis:",
                             [col for col in numeric_columns if col != x_col],
                             index=0, key="3d_y")
    with col3:
        remaining_cols = [col for col in numeric_columns if col != x_col and col != y_col]
        z_col = st.selectbox("Z-axis:", remaining_cols, index=0, key="3d_z")

    # Optional color by category and size by variable
    color_by = None
    size_by = None

    col1, col2 = st.columns(2)
    with col1:
        if categorical_columns:
            use_color = st.checkbox("Color by category", key="3d_use_color")
            if use_color:
                color_by = st.selectbox("Select category:", categorical_columns, key="3d_color")

    with col2:
        use_size = st.checkbox("Size by variable", key="3d_use_size")
        if use_size:
            size_options = [col for col in numeric_columns
                            if col != x_col and col != y_col and col != z_col]
            if size_options:
                size_by = st.selectbox("Size by:", size_options, key="3d_size")

    # Sample data for performance (3D plots are resource intensive)
    scatter_data = viz_data
    if len(scatter_data) > 1000:
        scatter_data = scatter_data.sample(1000, random_state=42)
        st.info("Using a sample of 1000 rows for better 3D plot performance.")

    try:
        # Create 3D scatter plot
        fig = px.scatter_3d(
            scatter_data,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_by,
            size=size_by,
            opacity=0.7,
            title="3D Scatter Plot"
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)
        return fig, f"3d_scatter_{x_col}_{y_col}_{z_col}"
    except Exception as e:
        st.error(f"Error creating 3D scatter plot: {str(e)}")
        return None, None


def display_sunburst_chart(viz_data, categorical_columns, numeric_columns=None):
    """
    Display a sunburst chart

    Args:
        viz_data: DataFrame containing the data
        categorical_columns: List of categorical column names
        numeric_columns: List of numeric column names (optional)
    """
    st.write("Sunburst Chart shows hierarchical relationships in categorical data")

    if len(categorical_columns) < 2:
        st.warning("Need at least 2 categorical columns for a sunburst chart.")
        return None, None

    # Select categorical columns for hierarchy
    selected_columns = st.multiselect(
        "Select categorical columns for hierarchy (order matters):",
        categorical_columns,
        default=categorical_columns[:min(len(categorical_columns), 3)],
        key="sunburst_cols"
    )

    if len(selected_columns) < 2:
        st.warning("Please select at least 2 categorical columns for the sunburst chart.")
        return None, None

    # Optional value column
    use_values = st.checkbox("Include values", key="sunburst_use_values")
    value_col = None

    if use_values and numeric_columns:
        value_col = st.selectbox("Select value column:", numeric_columns, key="sunburst_value")

    # Limit data if too many rows or too many categories
    sunburst_data = viz_data

    # Check if any column has too many categories
    show_warning = False
    for col in selected_columns:
        if viz_data[col].nunique() > 50:
            show_warning = True

    if show_warning:
        st.warning("Some columns have many unique values, which may affect chart readability.")

    if len(sunburst_data) > 1000:
        sunburst_data = sunburst_data.sample(1000, random_state=42)
        st.info("Using a sample of 1000 rows for better performance.")

    try:
        # Create sunburst chart
        fig = px.sunburst(
            sunburst_data,
            path=selected_columns,
            values=value_col,
            title="Sunburst Chart"
        )

        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        return fig, "sunburst_chart"
    except Exception as e:
        st.error(f"Error creating sunburst chart: {str(e)}")
        return None, None