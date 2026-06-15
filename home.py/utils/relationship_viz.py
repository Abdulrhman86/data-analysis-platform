import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_export import export_chart, export_matplotlib_chart


def display_scatter_plot(viz_data, numeric_columns, categorical_columns=None):
    """
    Display a scatter plot for the relationship between two or more numeric variables

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names (optional)
    """
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for a scatter plot.")
        return None, None

    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("Select X-axis column:", numeric_columns, key="scatter_x")
    with col2:
        y_columns = st.multiselect("Select Y-axis column(s):",
                                   [col for col in numeric_columns if col != x_column],
                                   default=[numeric_columns[1 if numeric_columns[0] == x_column else 0]],
                                   key="scatter_y")

    if not y_columns:
        st.warning("Please select at least one Y-axis column.")
        return None, None

    color_by = None
    if categorical_columns:
        use_color = st.checkbox("Color points by category", key="scatter_use_color")
        if use_color:
            color_by = st.selectbox("Select category for coloring:", categorical_columns,
                                    key="scatter_color")

    # Create scatter plot for each y-column
    fig = go.Figure()

    # Limit number of points for performance
    scatter_data = viz_data

    for y_col in y_columns:
        if color_by:
            for i, category in enumerate(scatter_data[color_by].unique()):
                if i > 20:  # Limit to 20 categories
                    st.warning(f"Too many categories in {color_by}. Showing only first 20.")
                    break

                subset = scatter_data[scatter_data[color_by] == category]
                fig.add_trace(go.Scatter(
                    x=subset[x_column],
                    y=subset[y_col],
                    mode='markers',
                    name=f"{y_col} - {category}",
                    marker=dict(size=8)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=scatter_data[x_column],
                y=scatter_data[y_col],
                mode='markers',
                name=y_col,
                marker=dict(size=8)
            ))

    fig.update_layout(
        title=f"Scatter Plot of {', '.join(y_columns)} vs {x_column}",
        xaxis_title=x_column,
        yaxis_title="Value",
        height=500,
        legend_title=color_by if color_by else "Variables"
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig, f"scatter_{x_column}"


def display_correlation_matrix(viz_data, numeric_columns):
    """
    Display a correlation matrix for numeric variables

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
    """
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return None, None

    # Options for correlation
    corr_method = st.radio(
        "Correlation Method",
        ["Pearson", "Spearman", "Kendall"],
        horizontal=True,
        key="corr_method"
    )

    # Select columns for correlation
    selected_columns = st.multiselect(
        "Select columns for correlation matrix:",
        numeric_columns,
        default=numeric_columns[:min(len(numeric_columns), 8)],
        key="corr_columns"
    )

    if len(selected_columns) < 2:
        st.warning("Please select at least 2 columns for correlation analysis.")
        return None, None

    # Calculate correlation
    corr = viz_data[selected_columns].corr(method=corr_method.lower())

    # Visualize correlation matrix
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f"{corr_method} Correlation Matrix"
    )
    fig.update_layout(height=max(400, min(800, len(selected_columns) * 30)))
    st.plotly_chart(fig, use_container_width=True)

    # Strongest correlations
    st.subheader("Strongest Correlations")

    # Create a DataFrame of correlations
    corr_df = corr.unstack().reset_index()
    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # Remove self-correlations
    corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]

    # Sort by absolute correlation value
    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs Correlation', ascending=False)

    # Display the top correlations
    st.dataframe(
        corr_df[['Variable 1', 'Variable 2', 'Correlation']].head(10),
        use_container_width=True
    )

    return fig, f"correlation_{corr_method}"


def display_pair_plot(viz_data, numeric_columns, categorical_columns=None):
    """
    Display a pair plot for multiple numeric variables

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names (optional)
    """
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for a pair plot.")
        return None, None

    # Select columns for pair plot with warning about performance
    st.warning("Pair plots can be computationally intensive. Limiting to 5 columns is recommended.")
    selected_columns = st.multiselect(
        "Select columns for pair plot:",
        numeric_columns,
        default=numeric_columns[:min(len(numeric_columns), 4)],
        key="pair_columns"
    )

    if len(selected_columns) < 2:
        st.warning("Please select at least 2 columns for pair plot.")
        return None, None
    elif len(selected_columns) > 7:
        st.warning(
            "Too many columns will make the pair plot slow and hard to read. Consider selecting fewer columns.")

    color_by = None
    if categorical_columns:
        use_color = st.checkbox("Color by category", key="pair_use_color")
        if use_color:
            color_by = st.selectbox("Select category for coloring:", categorical_columns, key="pair_color")
            # Limit categories for performance
            if viz_data[color_by].nunique() > 10:
                st.warning(f"Too many categories in {color_by}. Performance may be affected.")

    # Further sample data for pair plot if necessary
    pair_plot_data = viz_data
    if len(pair_plot_data) > 1000:
        pair_plot_data = pair_plot_data.sample(1000, random_state=42)
        st.info(f"Using a sample of 1000 rows for pair plot to improve performance.")

    with st.spinner("Generating pair plot... This may take a moment."):
        try:
            # Create pair plot with seaborn
            g = sns.pairplot(
                pair_plot_data,
                vars=selected_columns,
                hue=color_by,
                diag_kind="kde",
                plot_kws={"alpha": 0.6},
                height=2.5,  # Smaller subplot size for better performance
            )
            st.pyplot(g.fig)
            export_matplotlib_chart(g.fig, "pairplot")

            # Return None since we're using matplotlib and not returning a plotly figure
            return None, "pairplot"
        except Exception as e:
            st.error(f"Error generating pair plot: {str(e)}")
            return None, None
        finally:
            plt.close('all')  # Close all figures to free memory


def display_grouped_bar_chart(viz_data, categorical_columns):
    """
    Display a grouped bar chart for categorical variables

    Args:
        viz_data: DataFrame containing the data
        categorical_columns: List of categorical column names
    """
    if len(categorical_columns) < 2:
        st.warning("Need at least 2 categorical columns for a grouped bar chart.")
        return None, None

    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("Select main category (X-axis):", categorical_columns, key="grouped_bar_x")
    with col2:
        group_by = st.selectbox("Select grouping category:",
                                [col for col in categorical_columns if col != x_column],
                                key="grouped_bar_group")

    # Limit categories if too many
    x_unique = min(viz_data[x_column].nunique(), 20)
    group_unique = min(viz_data[group_by].nunique(), 10)

    if viz_data[x_column].nunique() > 20 or viz_data[group_by].nunique() > 10:
        st.warning(
            f"Too many categories. Limiting to top {x_unique} in {x_column} and top {group_unique} in {group_by}.")

        # Get top categories for each column
        top_x_cats = viz_data[x_column].value_counts().nlargest(x_unique).index
        top_group_cats = viz_data[group_by].value_counts().nlargest(group_unique).index

        # Filter data
        filtered_data = viz_data[viz_data[x_column].isin(top_x_cats) & viz_data[group_by].isin(top_group_cats)]
    else:
        filtered_data = viz_data

    # Calculate grouped counts
    grouped_counts = filtered_data.groupby([x_column, group_by]).size().reset_index(name='Count')

    # Create grouped bar chart
    fig = px.bar(
        grouped_counts,
        x=x_column,
        y='Count',
        color=group_by,
        barmode='group',
        title=f"Grouped Bar Chart: {x_column} by {group_by}"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    return fig, f"grouped_bar_{x_column}_{group_by}"


def display_grouped_box_plot(viz_data, numeric_columns, categorical_columns):
    """
    Display a grouped box plot for numeric and categorical variables

    Args:
        viz_data: DataFrame containing the data
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
    """
    if len(numeric_columns) < 1 or len(categorical_columns) < 1:
        st.warning("Need at least 1 numeric column and 1 categorical column for a grouped box plot.")
        return None, None

    col1, col2 = st.columns(2)
    with col1:
        numeric_col = st.selectbox("Select numeric variable:", numeric_columns, key="box_numeric")
    with col2:
        cat_col = st.selectbox("Select categorical variable:", categorical_columns, key="box_cat")

    # Limit categories if too many
    if viz_data[cat_col].nunique() > 15:
        st.warning(f"Too many categories in {cat_col}. Showing only top 15 by frequency.")
        top_cats = viz_data[cat_col].value_counts().nlargest(15).index
        filtered_data = viz_data[viz_data[cat_col].isin(top_cats)]
    else:
        filtered_data = viz_data

    # Optional second grouping
    use_second_group = st.checkbox("Add second grouping variable", key="box_use_second")
    second_group = None
    if use_second_group and len(categorical_columns) >= 2:
        second_group = st.selectbox(
            "Select second grouping variable:",
            [col for col in categorical_columns if col != cat_col],
            key="box_second_group"
        )

        # Further limit categories if using second grouping
        if second_group and viz_data[second_group].nunique() > 5:
            st.warning(f"Too many categories in second grouping. Showing only top 5 by frequency.")
            top_second_cats = viz_data[second_group].value_counts().nlargest(5).index
            filtered_data = filtered_data[filtered_data[second_group].isin(top_second_cats)]

    # Create box plot
    if second_group:
        fig = px.box(
            filtered_data,
            x=cat_col,
            y=numeric_col,
            color=second_group,
            title=f"Box Plot of {numeric_col} by {cat_col} and {second_group}"
        )
    else:
        fig = px.box(
            filtered_data,
            x=cat_col,
            y=numeric_col,
            title=f"Box Plot of {numeric_col} by {cat_col}"
        )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    return fig, f"grouped_box_{numeric_col}_{cat_col}"