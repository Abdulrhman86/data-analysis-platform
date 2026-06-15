import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_export import export_chart, export_matplotlib_chart


def display_numeric_variable(viz_data, selected_column, chart_type):
    """
    Display visualizations for a single numeric variable

    Args:
        viz_data: DataFrame containing the data
        selected_column: Name of the column to visualize
        chart_type: Type of chart to display (Histogram, Box Plot, etc.)
    """
    if chart_type == "Histogram":
        bins = st.slider("Number of bins:", 5, 100, 20, key="histogram_bins")

        fig = px.histogram(
            viz_data,
            x=selected_column,
            nbins=bins,
            title=f"Histogram of {selected_column}",
            opacity=0.7
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Add descriptive stats
        st.write("#### Descriptive Statistics:")
        stats = viz_data[selected_column].describe()
        st.dataframe(stats)

        return fig, f"histogram_{selected_column}"

    elif chart_type == "Box Plot":
        fig = px.box(
            viz_data,
            y=selected_column,
            title=f"Box Plot of {selected_column}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Show potential outliers
        q1 = viz_data[selected_column].quantile(0.25)
        q3 = viz_data[selected_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = viz_data[(viz_data[selected_column] < lower_bound) | (viz_data[selected_column] > upper_bound)]

        if not outliers.empty:
            st.write(f"#### Potential Outliers ({len(outliers)} found in sample):")
            st.dataframe(outliers.head(10) if len(outliers) > 10 else outliers)

        return fig, f"boxplot_{selected_column}"

    elif chart_type == "Violin Plot":
        fig = px.violin(
            viz_data,
            y=selected_column,
            box=True,
            points="all",
            title=f"Violin Plot of {selected_column}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        return fig, f"violin_{selected_column}"

    elif chart_type == "KDE Plot":
        # Use a more memory-efficient approach
        plt.figure(figsize=(10, 6))
        sns.kdeplot(viz_data[selected_column].dropna(), fill=True)
        plt.title(f"KDE Plot of {selected_column}")
        plt.grid(True, alpha=0.3)

        # Display the plot
        st.pyplot(plt)
        export_matplotlib_chart(plt, f"kde_{selected_column}", width=6, height=4)

        # Return None since we're using matplotlib and not returning a plotly figure
        return None, f"kde_{selected_column}"


def display_categorical_variable(viz_data, selected_column, chart_type):
    """
    Display visualizations for a single categorical variable

    Args:
        viz_data: DataFrame containing the data
        selected_column: Name of the column to visualize
        chart_type: Type of chart to display (Bar Chart, Pie Chart, etc.)
    """
    # Handle null values in display
    value_counts = viz_data[selected_column].value_counts(dropna=False)
    value_counts.index = value_counts.index.fillna("Missing/NaN")

    if chart_type == "Bar Chart":
        # Option to sort
        sort_by = st.radio("Sort by:", ["Value", "Frequency (Ascending)", "Frequency (Descending)"],
                           horizontal=True,
                           key="bar_sort")

        if sort_by == "Value":
            value_counts = value_counts.sort_index()
        elif sort_by == "Frequency (Ascending)":
            value_counts = value_counts.sort_values()
        else:  # Descending
            value_counts = value_counts.sort_values(ascending=False)

        # Limit categories if too many
        if len(value_counts) > 20:
            show_top = st.slider("Show top N categories:", 5, min(50, len(value_counts)), 20, key="bar_top_n")
            value_counts = value_counts.iloc[:show_top]

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Bar Chart of {selected_column}",
            labels={'x': selected_column, 'y': 'Count'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        return fig, f"bar_{selected_column}"

    elif chart_type == "Pie Chart":
        # Limit categories if too many
        if len(value_counts) > 10:
            show_top = st.slider("Show top N categories:", 5, min(20, len(value_counts)), 10, key="pie_top_n")
            remaining_count = value_counts.iloc[show_top:].sum()
            value_counts = value_counts.iloc[:show_top]
            if remaining_count > 0:
                value_counts["Other"] = remaining_count

        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {selected_column}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        return fig, f"pie_{selected_column}"

    elif chart_type == "Count Plot":
        # Limit categories if too many
        if len(value_counts) > 20:
            show_top = st.slider("Show top N categories:", 5, min(50, len(value_counts)), 20, key="count_top_n")
            categories_to_show = value_counts.iloc[:show_top].index
        else:
            categories_to_show = value_counts.index

        try:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=viz_data[selected_column], order=categories_to_show)
            plt.title(f"Count Plot of {selected_column}")
            plt.grid(True, alpha=0.3)

            # Display the plot
            st.pyplot(plt)
            export_matplotlib_chart(plt, f"countplot_{selected_column}")

            # Return None since we're using matplotlib and not returning a plotly figure
            return None, f"countplot_{selected_column}"
        finally:
            plt.close()  # Ensure the figure is closed

    elif chart_type == "Percentage Distribution":
        # Calculate percentages
        percentage = (value_counts / value_counts.sum() * 100).round(2)

        # Format for display
        df_display = pd.DataFrame({
            'Category': percentage.index,
            'Count': value_counts.values,
            'Percentage': percentage.values
        }).sort_values('Count', ascending=False)

        st.dataframe(df_display, use_container_width=True)

        # Create a horizontal bar chart of percentages
        fig = px.bar(
            df_display,
            y='Category',
            x='Percentage',
            orientation='h',
            text='Percentage',
            title=f"Percentage Distribution of {selected_column}"
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=max(400, min(800, len(df_display) * 20)))
        st.plotly_chart(fig, use_container_width=True)

        return fig, f"pct_dist_{selected_column}"