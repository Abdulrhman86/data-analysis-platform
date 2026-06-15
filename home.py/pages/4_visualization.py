import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing_utils import detect_column_types
from utils.chart_export import export_chart, export_matplotlib_chart
from config import Config  # Import Config for theme

# Import visualization utility functions
from utils.single_variable_viz import display_numeric_variable, display_categorical_variable
from utils.relationship_viz import (
    display_scatter_plot, display_correlation_matrix, display_pair_plot,
    display_grouped_bar_chart, display_grouped_box_plot
)
from utils.distribution_viz import (
    display_numeric_distributions, display_category_distribution, display_distribution_grid
)
from utils.time_series_viz import (
    display_time_series_plot, display_resampled_trends,
    display_seasonality_analysis, safe_convert_datetime
)
from utils.summary_statistics_viz import (
    display_numeric_summary, display_categorical_summary,
    display_group_statistics, display_missing_value_analysis
)
from utils.advanced_viz import (
    display_parallel_coordinates, display_radar_chart,
    display_3d_scatter, display_sunburst_chart
)

# Define a max sample size for performance
MAX_SAMPLE_SIZE = 5000


# Function to get a performance-optimized dataset
def get_optimized_dataset(df, max_rows=MAX_SAMPLE_SIZE):
    """Returns a sampled version of the dataset if it's too large"""
    if len(df) > max_rows:
        st.info(f"Dataset has {len(df)} rows. Using a sample of {max_rows} rows for visualizations.")
        return df.sample(max_rows, random_state=42)
    return df


# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Data Visualization"
)

# Apply the CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)

# Page header with back navigation
st.markdown('<div class="page-header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
    if st.button('← ', key='back_button', help="Return to preprocessing page"):
        st.switch_page("pages/3_preprocessing.py")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="margin: 0; color: white;">Data Visualization</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Check if data exists in session state
if 'data' not in st.session_state or st.session_state.data is None:
    st.markdown('<div class="card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
    st.warning("No data loaded. Please upload a dataset first.")
    if st.button('Go to Data Upload', use_container_width=True):
        st.switch_page("pages/1_upload_data.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Get the data and create a working copy
data = st.session_state.data.copy()
viz_data = get_optimized_dataset(data)

# Dataset Overview Card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

# Basic data info
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<span class="metric-label">Rows</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{data.shape[0]:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<span class="metric-label">Columns</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{data.shape[1]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<span class="metric-label">Missing Values</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{data.isna().sum().sum():,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Detect column types
column_types = detect_column_types(data)
numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
categorical_columns = [col for col, type_ in column_types.items() if type_ == 'categorical']
datetime_columns = [col for col, type_ in column_types.items() if type_ == 'datetime']

# Display column type counts
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<span class="metric-label">Numeric Columns</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(numeric_columns)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<span class="metric-label">Categorical Columns</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(categorical_columns)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<span class="metric-label">Datetime Columns</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(datetime_columns)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Visualization card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Visualization</div>', unsafe_allow_html=True)

# Initialize session state for current_fig if it doesn't exist
if 'current_fig' not in st.session_state:
    st.session_state.current_fig = None
if 'current_chart_name' not in st.session_state:
    st.session_state.current_chart_name = None

# Create tabs for different visualization types
viz_tabs = st.tabs([
    "Single Variable", "Relationship", "Distribution",
    "Time Series", "Summary Statistics", "Advanced"
])


def add_to_dashboard(fig, chart_name, chart_type):
    """
    Add the current visualization to a dashboard

    Args:
        fig: The plotly figure object
        chart_name: Name of the chart
        chart_type: Type of chart (e.g., 'Histogram', 'Box Plot', 'Scatter Plot')
    """
    import uuid
    # Generate a truly unique ID for this specific add operation
    unique_id = str(uuid.uuid4())[:8]

    # Create a comprehensive mapping from visualization chart types to dashboard chart types
    chart_type_mapping = {
        # Single Variable - Numeric
        "Histogram": "histogram",
        "Box Plot": "box",
        "Violin Plot": "violin",
        "KDE Plot": "kde",

        # Single Variable - Categorical
        "Bar Chart": "bar",
        "Pie Chart": "pie",
        "Count Plot": "bar",
        "Percentage Distribution": "bar",

        # Relationship
        "Scatter Plot": "scatter",
        "Correlation Matrix": "heatmap",
        "Pair Plot": "custom_grid",
        "Grouped Bar Chart": "bar",
        "Grouped Box Plot": "box",

        # Distribution
        "Compare Numeric Distributions": "histogram",
        "Category Distribution in Groups": "heatmap",
        "Distribution Grid": "custom_grid",

        # Time Series
        "Time Series Plot": "line",
        "Resampled Trends": "line",
        "Seasonality Analysis": "custom_time",

        # Summary Statistics
        "Numeric Summary": "custom_summary",
        "Categorical Summary": "bar",
        "Group Statistics": "bar",
        "Missing Value Analysis": "bar",

        # Advanced
        "Parallel Coordinates": "custom_parallel",
        "Radar Chart": "custom_radar",
        "3D Scatter Plot": "custom_3d",
        "Sunburst Chart": "custom_sunburst"
    }

    # Get the base chart type for dashboard
    dashboard_chart_type = chart_type_mapping.get(chart_type, "custom")

    # Check if dashboards exist
    if 'dashboards' not in st.session_state:
        st.session_state.dashboards = {}

    if not st.session_state.dashboards:
        st.warning("No dashboards available. Create a dashboard first.")
        return

    with st.form(key=f"add_to_dashboard_form_{unique_id}"):
        st.subheader("Add to Dashboard")

        # Select dashboard
        dashboard_name = st.selectbox(
            "Select dashboard:",
            options=list(st.session_state.dashboards.keys()),
            key=f"dashboard_select_{unique_id}"
        )

        # Component settings
        component_title = st.text_input("Component title:", chart_name,
                                        key=f"title_{unique_id}")

        # Get available columns for configuration
        df_columns = viz_data.columns.tolist()
        numeric_columns = [col for col in df_columns if viz_data[col].dtype in ['int64', 'float64']]
        categorical_columns = [col for col in df_columns if col not in numeric_columns]

        # Create appropriate configuration based on chart type
        config = {}

        # Configure different chart types
        if dashboard_chart_type == "histogram":
            config['x'] = st.selectbox(
                "Select column for histogram:",
                options=df_columns,
                key=f"hist_x_{unique_id}"
            )
            config['bins'] = st.slider(
                "Number of bins:",
                min_value=5,
                max_value=100,
                value=20,
                key=f"hist_bins_{unique_id}"
            )

        elif dashboard_chart_type == "bar":
            config['x'] = st.selectbox(
                "X-axis column:",
                options=df_columns,
                key=f"bar_x_{unique_id}"
            )
            if numeric_columns:
                config['y'] = st.selectbox(
                    "Y-axis column:",
                    options=numeric_columns,
                    key=f"bar_y_{unique_id}"
                )
            config['orientation'] = st.radio(
                "Orientation:",
                options=["vertical", "horizontal"],
                key=f"bar_orient_{unique_id}"
            )

        elif dashboard_chart_type == "scatter":
            if len(numeric_columns) >= 2:
                config['x'] = st.selectbox(
                    "X-axis column:",
                    options=numeric_columns,
                    key=f"scatter_x_{unique_id}"
                )
                y_options = [col for col in numeric_columns if col != config['x']]
                config['y'] = st.selectbox(
                    "Y-axis column:",
                    options=y_options,
                    key=f"scatter_y_{unique_id}"
                )

                if categorical_columns:
                    use_color = st.checkbox("Color by category", key=f"use_color_{unique_id}")
                    if use_color:
                        config['color'] = st.selectbox(
                            "Color by:",
                            options=categorical_columns,
                            key=f"scatter_color_{unique_id}"
                        )

        elif dashboard_chart_type == "line":
            config['x'] = st.selectbox(
                "X-axis column:",
                options=df_columns,
                key=f"line_x_{unique_id}"
            )
            if numeric_columns:
                config['y'] = st.selectbox(
                    "Y-axis column:",
                    options=numeric_columns,
                    key=f"line_y_{unique_id}"
                )

        elif dashboard_chart_type == "box":
            if numeric_columns:
                config['y'] = st.selectbox(
                    "Value column:",
                    options=numeric_columns,
                    key=f"box_y_{unique_id}"
                )

            if categorical_columns:
                use_category = st.checkbox("Group by category", key=f"use_box_cat_{unique_id}")
                if use_category:
                    config['x'] = st.selectbox(
                        "Group by:",
                        options=categorical_columns,
                        key=f"box_x_{unique_id}"
                    )

        elif dashboard_chart_type == "violin":
            if numeric_columns:
                config['y'] = st.selectbox(
                    "Value column:",
                    options=numeric_columns,
                    key=f"violin_y_{unique_id}"
                )

            if categorical_columns:
                use_category = st.checkbox("Group by category", key=f"use_violin_cat_{unique_id}")
                if use_category:
                    config['x'] = st.selectbox(
                        "Group by:",
                        options=categorical_columns,
                        key=f"violin_x_{unique_id}"
                    )

        elif dashboard_chart_type == "pie":
            if numeric_columns:
                config['values'] = st.selectbox(
                    "Values column:",
                    options=numeric_columns,
                    key=f"pie_values_{unique_id}"
                )

            if categorical_columns:
                config['names'] = st.selectbox(
                    "Names column:",
                    options=categorical_columns,
                    key=f"pie_names_{unique_id}"
                )

        elif dashboard_chart_type == "heatmap":
            if len(categorical_columns) >= 2:
                config['x'] = st.selectbox(
                    "X-axis column:",
                    options=categorical_columns,
                    key=f"heatmap_x_{unique_id}"
                )
                x_options = [col for col in categorical_columns if col != config['x']]
                config['y'] = st.selectbox(
                    "Y-axis column:",
                    options=x_options,
                    key=f"heatmap_y_{unique_id}"
                )

            if numeric_columns:
                config['values'] = st.selectbox(
                    "Values column:",
                    options=numeric_columns,
                    key=f"heatmap_values_{unique_id}"
                )

        # Layout settings
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider(
                "Width (columns):",
                min_value=1,
                max_value=12,
                value=6,
                key=f"width_{unique_id}"
            )
        with col2:
            height = st.slider(
                "Height (pixels):",
                min_value=200,
                max_value=800,
                value=400,
                key=f"height_{unique_id}"
            )

        submitted = st.form_submit_button("Add to Dashboard")

    if submitted:
        from utils.dashboard_module import DashboardComponent

        # Create component
        component = DashboardComponent(
            title=component_title,
            chart_type=dashboard_chart_type,
            config=config,
            width=width,
            height=height,
            source='visualization'
        )

        # Add to dashboard
        dashboard = st.session_state.dashboards[dashboard_name]
        dashboard.add_component(component)

        st.success(f"Added '{component_title}' to dashboard: {dashboard_name}")


# Tab 1: Single Variable Analysis
with viz_tabs[0]:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Single Variable Analysis</h3>', unsafe_allow_html=True)

    # Select variable type
    var_type = st.radio(
        "Variable Type",
        ["Numeric", "Categorical"],
        horizontal=True
    )

    if var_type == "Numeric" and numeric_columns:
        # Numeric variable visualization
        selected_column = st.selectbox("Select a numeric column:", numeric_columns, key="single_var_numeric")

        chart_type = st.radio(
            "Chart Type",
            ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"],
            horizontal=True,
            key="single_var_chart_type"
        )

        # Display the selected visualization
        fig, chart_name = display_numeric_variable(viz_data, selected_column, chart_type)

        # Save the current figure and chart name
        if fig is not None:
            st.session_state.current_fig = fig
            st.session_state.current_chart_name = chart_name

            # Export chart button (moved to the bottom of each tab)
            with st.expander("Export Options"):
                export_chart(fig, chart_name)

                st.markdown("---")
                st.subheader("Add to Dashboard")
                add_to_dashboard(fig, chart_name, chart_type)

    elif var_type == "Categorical" and categorical_columns:
        # Categorical variable visualization
        selected_column = st.selectbox("Select a categorical column:", categorical_columns, key="single_var_cat")

        chart_type = st.radio(
            "Chart Type",
            ["Bar Chart", "Pie Chart", "Count Plot", "Percentage Distribution"],
            horizontal=True,
            key="single_var_cat_chart"
        )

        # Display the selected visualization
        fig, chart_name = display_categorical_variable(viz_data, selected_column, chart_type)

        # Save the current figure and chart name
        if fig is not None:
            st.session_state.current_fig = fig
            st.session_state.current_chart_name = chart_name

            # Export chart button
            with st.expander("Export Options"):
                export_chart(fig, chart_name)

                st.markdown("---")
                st.subheader("Add to Dashboard")
                add_to_dashboard(fig, chart_name, chart_type)
    else:
        st.warning("No columns of the selected type found in the dataset.")

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Relationship Analysis
with viz_tabs[1]:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Relationship Analysis</h3>', unsafe_allow_html=True)

    chart_type = st.selectbox(
        "Chart Type",
        ["Scatter Plot", "Correlation Matrix", "Pair Plot", "Grouped Bar Chart", "Grouped Box Plot"],
        key="relationship_chart_type"
    )

    if chart_type == "Scatter Plot":
        fig, chart_name = display_scatter_plot(viz_data, numeric_columns, categorical_columns)
    elif chart_type == "Correlation Matrix":
        fig, chart_name = display_correlation_matrix(viz_data, numeric_columns)
    elif chart_type == "Pair Plot":
        fig, chart_name = display_pair_plot(viz_data, numeric_columns, categorical_columns)
    elif chart_type == "Grouped Bar Chart":
        fig, chart_name = display_grouped_bar_chart(viz_data, categorical_columns)
    elif chart_type == "Grouped Box Plot":
        fig, chart_name = display_grouped_box_plot(viz_data, numeric_columns, categorical_columns)

    # Save the current figure and chart name
    if fig is not None:
        st.session_state.current_fig = fig
        st.session_state.current_chart_name = chart_name

        # Export chart button
        with st.expander("Export Options"):
            export_chart(fig, chart_name)

            st.markdown("---")
            st.subheader("Add to Dashboard")
            add_to_dashboard(fig, chart_name, chart_type)

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Distribution Analysis
with viz_tabs[2]:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Distribution Analysis</h3>', unsafe_allow_html=True)

    analysis_type = st.radio(
        "Analysis Type",
        ["Compare Numeric Distributions", "Category Distribution in Groups", "Distribution Grid"],
        horizontal=True,
        key="dist_type"
    )

    if analysis_type == "Compare Numeric Distributions":
        fig, chart_name = display_numeric_distributions(viz_data, numeric_columns)
    elif analysis_type == "Category Distribution in Groups":
        fig, chart_name = display_category_distribution(viz_data, categorical_columns)
    elif analysis_type == "Distribution Grid":
        fig, chart_name = display_distribution_grid(viz_data, numeric_columns, categorical_columns)

    # Save the current figure and chart name
    if fig is not None:
        st.session_state.current_fig = fig
        st.session_state.current_chart_name = chart_name

        # Export chart button
        with st.expander("Export Options"):
            export_chart(fig, chart_name)

            st.markdown("---")
            st.subheader("Add to Dashboard")
            add_to_dashboard(fig, chart_name, analysis_type)

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Time Series Analysis
with viz_tabs[3]:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Time Series Analysis</h3>', unsafe_allow_html=True)

    if not datetime_columns:
        st.warning(
            "No datetime columns detected in the dataset. Please preprocess your data to convert date/time columns.")
    else:
        # Select datetime column
        date_col = st.selectbox("Select date/time column:", datetime_columns, key="ts_date_col")

        # Check if column is properly parsed as datetime
        if not pd.api.types.is_datetime64_dtype(viz_data[date_col]):
            st.warning(f"Column {date_col} is not properly parsed as datetime. Attempting to convert.")
            # Use safe conversion function
            converted_data, error = safe_convert_datetime(viz_data, date_col)
            if error:
                st.error(error)
                st.stop()
            else:
                viz_data = converted_data
                st.success("Successfully converted datetime column.")

        # Select analysis type
        analysis_type = st.radio(
            "Analysis Type",
            ["Time Series Plot", "Resampled Trends", "Seasonality Analysis"],
            horizontal=True,
            key="ts_analysis_type"
        )

        if analysis_type == "Time Series Plot":
            # Select variables to plot
            value_cols = st.multiselect(
                "Select numeric columns to plot over time:",
                numeric_columns,
                default=numeric_columns[:1] if numeric_columns else [],
                key="ts_value_cols"
            )

            fig, chart_name = display_time_series_plot(viz_data, date_col, value_cols, categorical_columns)

        elif analysis_type == "Resampled Trends":
            # Select variable for trend analysis
            value_col = st.selectbox("Select numeric column to analyze:", numeric_columns, key="resample_value")

            fig, chart_name = display_resampled_trends(viz_data, date_col, value_col)

        elif analysis_type == "Seasonality Analysis":
            # Select variable for seasonality analysis
            value_col = st.selectbox("Select numeric column to analyze:", numeric_columns, key="season_value")

            fig, chart_name = display_seasonality_analysis(viz_data, date_col, value_col)

        # Save the current figure and chart name
        if fig is not None:
            st.session_state.current_fig = fig
            st.session_state.current_chart_name = chart_name

            # Export chart button
            with st.expander("Export Options"):
                export_chart(fig, chart_name)

                st.markdown("---")
                st.subheader("Add to Dashboard")
                add_to_dashboard(fig, chart_name, analysis_type)

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: Summary Statistics
with viz_tabs[4]:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Summary Statistics</h3>', unsafe_allow_html=True)

    stats_type = st.radio(
        "Statistics Type",
        ["Numeric Summary", "Categorical Summary", "Group Statistics", "Missing Value Analysis"],
        horizontal=True,
        key="stats_type"
    )

    if stats_type == "Numeric Summary":
        fig, chart_name = display_numeric_summary(viz_data, numeric_columns)
    elif stats_type == "Categorical Summary":
        fig, chart_name = display_categorical_summary(viz_data, categorical_columns)
    elif stats_type == "Group Statistics":
        fig, chart_name = display_group_statistics(viz_data, numeric_columns, categorical_columns)
    elif stats_type == "Missing Value Analysis":
        fig, chart_name = display_missing_value_analysis(data)

    # Save the current figure and chart name
    if fig is not None:
        st.session_state.current_fig = fig
        st.session_state.current_chart_name = chart_name

        # Export chart button
        with st.expander("Export Options"):
            export_chart(fig, chart_name)

            st.markdown("---")
            st.subheader("Add to Dashboard")
            add_to_dashboard(fig, chart_name, stats_type)

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 6: Advanced Visualizations
with viz_tabs[5]:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Advanced Visualizations</h3>', unsafe_allow_html=True)

    viz_option = st.selectbox(
        "Select Visualization",
        ["Parallel Coordinates", "Radar Chart", "3D Scatter Plot", "Sunburst Chart"],
        key="adv_viz_type"
    )

    if viz_option == "Parallel Coordinates":
        fig, chart_name = display_parallel_coordinates(viz_data, numeric_columns, categorical_columns)
    elif viz_option == "Radar Chart":
        fig, chart_name = display_radar_chart(viz_data, numeric_columns, categorical_columns)
    elif viz_option == "3D Scatter Plot":
        fig, chart_name = display_3d_scatter(viz_data, numeric_columns, categorical_columns)
    elif viz_option == "Sunburst Chart":
        fig, chart_name = display_sunburst_chart(viz_data, categorical_columns, numeric_columns)

    # Save the current figure and chart name
    if fig is not None:
        st.session_state.current_fig = fig
        st.session_state.current_chart_name = chart_name

        # Export chart button
        with st.expander("Export Options"):
            export_chart(fig, chart_name)

            st.markdown("---")
            st.subheader("Add to Dashboard")
            add_to_dashboard(fig, chart_name, viz_option)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Navigation at the bottom
st.markdown('<div style="display: flex; justify-content: space-between; margin-top: 2rem;">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button('← Previous: Preprocessing', key='prev_button', use_container_width=True):
        st.switch_page("pages/3_preprocessing.py")
with col2:
    if st.button('Next: Dashboards →', key='next_button', use_container_width=True):
        st.switch_page("pages/5_dashboard.py")
st.markdown('</div>', unsafe_allow_html=True)