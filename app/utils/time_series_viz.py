import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.chart_export import export_chart


def safe_convert_datetime(df, column):
    """
    Safely convert a column to datetime without modifying original data

    Args:
        df: DataFrame containing the data
        column: Name of the column to convert

    Returns:
        tuple: (converted_dataframe, error_message)
    """
    temp_df = df.copy()
    try:
        temp_df[column] = pd.to_datetime(temp_df[column], errors='coerce')
        if temp_df[column].isna().sum() > 0.5 * len(temp_df):
            return None, "Too many conversion errors. Please check the format of your date column."
        return temp_df, None
    except Exception as e:
        return None, f"Error converting to datetime: {str(e)}"


def display_time_series_plot(viz_data, date_col, value_cols, categorical_columns=None):
    """
    Display a time series plot

    Args:
        viz_data: DataFrame containing the data
        date_col: Name of the date/time column
        value_cols: List of numeric columns to plot
        categorical_columns: List of categorical column names (optional)
    """
    if not value_cols:
        st.warning("Please select at least one numeric column to plot.")
        return None, None

    # Sort data by date
    plot_data = viz_data.sort_values(by=date_col)

    # Create plot
    fig = go.Figure()

    for col in value_cols:
        fig.add_trace(go.Scatter(
            x=plot_data[date_col],
            y=plot_data[col],
            mode='lines',
            name=col
        ))

    # Customize layout
    fig.update_layout(
        title=f"Time Series Plot",
        xaxis_title=date_col,
        yaxis_title="Value",
        height=500,
        legend_title="Variables"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Option to add a category filter
    if categorical_columns:
        use_filter = st.checkbox("Filter by category", key="ts_use_filter")

        if use_filter:
            filter_col = st.selectbox("Select category to filter by:", categorical_columns,
                                      key="ts_filter_col")

            # Limit categories if too many
            if viz_data[filter_col].nunique() > 10:
                st.warning(f"Too many categories in {filter_col}. Showing only top 10.")
                categories = viz_data[filter_col].value_counts().nlargest(10).index
            else:
                categories = viz_data[filter_col].unique()

            selected_categories = st.multiselect(
                "Select categories to display:",
                categories,
                default=categories[:min(len(categories), 3)],
                key="ts_categories"
            )

            if selected_categories:
                value_col = st.selectbox(
                    "Select variable to display:",
                    value_cols if len(value_cols) > 1 else value_cols,
                    key="ts_filtered_value"
                )

                # Create filtered plot
                filtered_fig = go.Figure()

                for category in selected_categories:
                    category_data = viz_data[viz_data[filter_col] == category].sort_values(by=date_col)

                    filtered_fig.add_trace(go.Scatter(
                        x=category_data[date_col],
                        y=category_data[value_col],
                        mode='lines',
                        name=str(category)
                    ))

                # Customize layout
                filtered_fig.update_layout(
                    title=f"{value_col} Over Time by {filter_col}",
                    xaxis_title=date_col,
                    yaxis_title=value_col,
                    height=500,
                    legend_title=filter_col
                )

                st.plotly_chart(filtered_fig, use_container_width=True)
                return filtered_fig, f"timeseries_filtered_{date_col}"

    return fig, f"timeseries_{date_col}"


def display_resampled_trends(viz_data, date_col, value_col):
    """
    Display resampled time series data

    Args:
        viz_data: DataFrame containing the data
        date_col: Name of the date/time column
        value_col: Name of the numeric column to analyze
    """
    # Select resampling frequency
    freq = st.selectbox(
        "Select resampling frequency:",
        ["Day", "Week", "Month", "Quarter", "Year"],
        key="resample_freq"
    )

    freq_map = {
        "Day": "D",
        "Week": "W",
        "Month": "ME",
        "Quarter": "QE",
        "Year": "YE"
    }

    # Select aggregation method
    agg_method = st.selectbox(
        "Select aggregation method:",
        ["Mean", "Sum", "Min", "Max", "Count"],
        key="resample_agg"
    )

    agg_map = {
        "Mean": "mean",
        "Sum": "sum",
        "Min": "min",
        "Max": "max",
        "Count": "count"
    }

    # Resample the data
    try:
        # Make sure data is sorted by date
        temp_data = viz_data.sort_values(by=date_col)

        # Set date as index for resampling
        temp_data = temp_data.set_index(date_col)

        # Resample
        resampled = temp_data[value_col].resample(freq_map[freq]).agg(agg_map[agg_method])

        # Reset index for plotting
        resampled = resampled.reset_index()

        # Create plot
        fig = px.line(
            resampled,
            x=date_col,
            y=value_col,
            title=f"{agg_method} of {value_col} ({freq} Frequency)"
        )

        # Add markers
        fig.update_traces(mode='lines+markers')

        # Customize layout
        fig.update_layout(
            xaxis_title=f"{freq}",
            yaxis_title=f"{agg_method} of {value_col}",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display the resampled data
        st.subheader("Resampled Data")
        st.dataframe(resampled, use_container_width=True)

        return fig, f"resampled_{value_col}_{freq}"

    except Exception as e:
        st.error(f"Error in resampling: {str(e)}")
        return None, None


def display_seasonality_analysis(viz_data, date_col, value_col):
    """
    Display seasonality analysis for time series data

    Args:
        viz_data: DataFrame containing the data
        date_col: Name of the date/time column
        value_col: Name of the numeric column to analyze
    """
    try:
        # Make copy of data to avoid modifying original
        temp_data = viz_data.copy()

        # Extract components
        temp_data['year'] = temp_data[date_col].dt.year
        temp_data['month'] = temp_data[date_col].dt.month
        temp_data['day'] = temp_data[date_col].dt.day
        temp_data['weekday'] = temp_data[date_col].dt.weekday
        temp_data['quarter'] = temp_data[date_col].dt.quarter

        # Select component for analysis
        component = st.selectbox(
            "Group by:",
            ["Month", "Day of Week", "Quarter", "Hour (if available)"],
            key="season_component"
        )

        if component == "Month":
            # Group by month
            monthly_data = temp_data.groupby('month')[value_col].agg(['mean', 'count'])
            monthly_data.columns = ['Average', 'Count']
            monthly_data = monthly_data.reset_index()

            # Convert month number to name
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_data['month_name'] = monthly_data['month'].apply(lambda x: month_names[x - 1])

            # Create plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(
                    x=monthly_data['month_name'],
                    y=monthly_data['Count'],
                    name='Count',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly_data['month_name'],
                    y=monthly_data['Average'],
                    name='Average',
                    mode='lines+markers',
                    marker=dict(size=10),
                    line=dict(width=3)
                ),
                secondary_y=False
            )

            # Customize layout
            fig.update_layout(
                title=f"Monthly Pattern for {value_col}",
                xaxis_title="Month",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )

            fig.update_yaxes(title_text="Average Value", secondary_y=False)
            fig.update_yaxes(title_text="Count", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
            return fig, f"seasonality_month_{value_col}"

        elif component == "Day of Week":
            # Group by weekday
            weekday_data = temp_data.groupby('weekday')[value_col].agg(['mean', 'count'])
            weekday_data.columns = ['Average', 'Count']
            weekday_data = weekday_data.reset_index()

            # Convert weekday number to name
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                             'Friday', 'Saturday', 'Sunday']
            weekday_data['weekday_name'] = weekday_data['weekday'].apply(lambda x: weekday_names[x])

            # Create plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(
                    x=weekday_data['weekday_name'],
                    y=weekday_data['Count'],
                    name='Count',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=weekday_data['weekday_name'],
                    y=weekday_data['Average'],
                    name='Average',
                    mode='lines+markers',
                    marker=dict(size=10),
                    line=dict(width=3)
                ),
                secondary_y=False
            )

            # Customize layout
            fig.update_layout(
                title=f"Day of Week Pattern for {value_col}",
                xaxis_title="Day of Week",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )

            fig.update_yaxes(title_text="Average Value", secondary_y=False)
            fig.update_yaxes(title_text="Count", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
            return fig, f"seasonality_weekday_{value_col}"

        elif component == "Quarter":
            # Group by quarter
            quarter_data = temp_data.groupby('quarter')[value_col].agg(['mean', 'count'])
            quarter_data.columns = ['Average', 'Count']
            quarter_data = quarter_data.reset_index()

            # Convert to quarter name
            quarter_data['quarter_name'] = 'Q' + quarter_data['quarter'].astype(str)

            # Create plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(
                    x=quarter_data['quarter_name'],
                    y=quarter_data['Count'],
                    name='Count',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=quarter_data['quarter_name'],
                    y=quarter_data['Average'],
                    name='Average',
                    mode='lines+markers',
                    marker=dict(size=10),
                    line=dict(width=3)
                ),
                secondary_y=False
            )

            # Customize layout
            fig.update_layout(
                title=f"Quarterly Pattern for {value_col}",
                xaxis_title="Quarter",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )

            fig.update_yaxes(title_text="Average Value", secondary_y=False)
            fig.update_yaxes(title_text="Count", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
            return fig, f"seasonality_quarter_{value_col}"

        elif component == "Hour (if available)":
            # Check if time component exists
            if hasattr(temp_data[date_col].dt, 'hour'):
                temp_data['hour'] = temp_data[date_col].dt.hour

                # Group by hour
                hourly_data = temp_data.groupby('hour')[value_col].agg(['mean', 'count'])
                hourly_data.columns = ['Average', 'Count']
                hourly_data = hourly_data.reset_index()

                # Create plot
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Bar(
                        x=hourly_data['hour'],
                        y=hourly_data['Count'],
                        name='Count',
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    secondary_y=True
                )

                fig.add_trace(
                    go.Scatter(
                        x=hourly_data['hour'],
                        y=hourly_data['Average'],
                        name='Average',
                        mode='lines+markers',
                        marker=dict(size=10),
                        line=dict(width=3)
                    ),
                    secondary_y=False
                )

                # Customize layout
                fig.update_layout(
                    title=f"Hourly Pattern for {value_col}",
                    xaxis_title="Hour of Day",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500
                )

                fig.update_yaxes(title_text="Average Value", secondary_y=False)
                fig.update_yaxes(title_text="Count", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)
                return fig, f"seasonality_hour_{value_col}"
            else:
                st.warning("No hour information available in the datetime column.")
                return None, None

    except Exception as e:
        st.error(f"Error in seasonality analysis: {str(e)}")
        return None, None