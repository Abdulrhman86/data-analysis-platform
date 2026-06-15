import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import base64
import io
import os
from datetime import datetime
import plotly.graph_objects as go
from utils import charts







class DashboardComponent:
    """Class representing a single component in a dashboard"""

    def __init__(self,
                 title,
                 chart_type,
                 config=None,
                 component_id=None,
                 width=6,
                 height=400,
                 source=None):
        """
        Initialize a dashboard component

        Args:
            title: Title of the component
            chart_type: Type of chart (e.g., 'bar', 'line', 'scatter')
            config: Configuration dictionary for the chart
            component_id: Unique identifier for the component
            width: Width in bootstrap columns (1-12)
            height: Height in pixels
            source: Source of the component (e.g., 'visualization', 'dashboard')
        """
        self.title = title
        self.chart_type = chart_type
        self.config = config or {}
        self.component_id = component_id or f"component_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.width = width
        self.height = height
        self.source = source or 'dashboard'  # Track where the component came from

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'title': self.title,
            'chart_type': self.chart_type,
            'config': self.config,
            'component_id': self.component_id,
            'width': self.width,
            'height': self.height,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data):
        """Create a DashboardComponent from a dictionary"""
        return cls(
            title=data['title'],
            chart_type=data['chart_type'],
            config=data.get('config', {}),
            component_id=data.get('component_id'),
            width=data.get('width', 6),
            height=data.get('height', 400),
            source=data.get('source', 'dashboard')
        )

    def render(self, df):
        """Render the component with data"""
        with st.container(border=True):
            st.subheader(self.title)

            try:
                if self.chart_type == 'bar':
                    self._render_bar_chart(df)
                elif self.chart_type == 'line':
                    self._render_line_chart(df)
                elif self.chart_type == 'scatter':
                    self._render_scatter_chart(df)
                elif self.chart_type == 'pie':
                    self._render_pie_chart(df)
                elif self.chart_type == 'histogram':
                    self._render_histogram(df)
                elif self.chart_type == 'box':
                    self._render_box_plot(df)
                elif self.chart_type == 'violin':
                    self._render_violin_plot(df)
                elif self.chart_type == 'kde':
                    self._render_kde_plot(df)
                elif self.chart_type == 'heatmap':
                    self._render_heatmap(df)
                elif self.chart_type == 'custom_parallel':
                    self._render_custom_parallel(df)
                elif self.chart_type == 'custom_radar':
                    self._render_custom_radar(df)
                elif self.chart_type == 'custom_3d':
                    self._render_custom_3d(df)
                elif self.chart_type == 'custom_sunburst':
                    self._render_custom_sunburst(df)
                elif self.chart_type == 'custom_grid':
                    self._render_custom_grid(df)
                elif self.chart_type == 'custom_time':
                    self._render_custom_time(df)
                elif self.chart_type == 'custom_summary':
                    self._render_custom_summary(df)
                else:
                    st.warning(f"Unknown chart type: {self.chart_type}")
                    st.json(self.config)  # Show config for debugging
            except Exception as e:
                st.error(f"Error rendering {self.chart_type} chart: {str(e)}")
                st.write("Configuration:", self.config)

    def get_figure(self, df):
        """Return this component's Plotly figure for the common chart types, else None.

        Used by PDF export; custom chart types (radar/3d/sunburst/...) return None.
        """
        builder = charts.BUILDERS.get(self.chart_type)
        if builder is None:
            return None
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            return builder(df, cfg)
        except Exception:
            return None

    def _render_violin_plot(self, df):
        """Render a violin plot component"""
        if not self.config.get('y'):
            st.error("Missing required configuration: y")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_violin(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering violin plot: {str(e)}")

    def _render_kde_plot(self, df):
        """Render a KDE plot component"""
        if not self.config.get('x'):
            st.error("Missing required configuration: x")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_kde(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering KDE plot: {str(e)}")

    def _render_box_plot(self, df):
        """Render a box plot component"""
        if not self.config.get('y'):
            st.error("Missing required configuration: y")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_box(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering box plot: {str(e)}")

    def _render_custom_parallel(self, df):
        """Render a parallel coordinates plot"""
        # Get dimensions from config or use numeric columns
        dims = self.config.get('dimensions')
        if not dims:
            dims = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])][:5]

        if not dims:
            st.error("No numeric columns available for parallel coordinates plot")
            return

        # Validate columns exist
        valid_dims = [dim for dim in dims if dim in df.columns]
        if not valid_dims:
            st.error(
                f"No valid columns for parallel coordinates. Requested: {dims}, Available: {', '.join(df.columns)}")
            return

        # Create chart
        try:
            fig = px.parallel_coordinates(
                df,
                dimensions=valid_dims,
                title=self.title,
                height=self.height
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering parallel coordinates: {str(e)}")

    def _render_custom_radar(self, df):
        """Render a radar chart"""
        category = self.config.get('category')
        variables = self.config.get('variables')

        if not category or not variables:
            st.error("Missing required configuration: category and variables")
            st.json(self.config)
            return

        # Validate columns exist
        missing_cols = []
        if category not in df.columns:
            missing_cols.append(category)

        for var in variables:
            if var not in df.columns:
                missing_cols.append(var)

        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}")
            return

        # Create chart
        try:
            # Calculate mean values for each category and variable
            radar_data = df.groupby(category)[variables].mean()

            # Normalize data for radar chart (0-1 scale)
            radar_data_norm = radar_data.copy()
            for col in variables:
                min_val = radar_data[col].min()
                max_val = radar_data[col].max()
                if max_val > min_val:  # Avoid division by zero
                    radar_data_norm[col] = (radar_data[col] - min_val) / (max_val - min_val)

            # Create radar chart
            fig = go.Figure()

            for category_name in radar_data_norm.index:
                fig.add_trace(go.Scatterpolar(
                    r=radar_data_norm.loc[category_name].values.tolist() + [
                        radar_data_norm.loc[category_name].values[0]],
                    theta=variables + [variables[0]],
                    fill='toself',
                    name=str(category_name)
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=self.title,
                height=self.height
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering radar chart: {str(e)}")

    def _render_custom_3d(self, df):
        """Render a 3D scatter plot"""
        x = self.config.get('x')
        y = self.config.get('y')
        z = self.config.get('z')
        color = self.config.get('color')

        if not x or not y or not z:
            st.error("Missing required configuration: x, y, and z")
            return

        # Validate columns exist
        missing_cols = []
        for col in [x, y, z]:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}")
            return

        if color and color not in df.columns:
            st.warning(f"Color column '{color}' not found. Color coding will be ignored.")
            color = None

        # Create chart
        try:
            fig = px.scatter_3d(
                df,
                x=x,
                y=y,
                z=z,
                color=color,
                title=self.title,
                height=self.height
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering 3D scatter plot: {str(e)}")

    def _render_custom_sunburst(self, df):
        """Render a sunburst chart"""
        path = self.config.get('path')
        values = self.config.get('values')

        if not path:
            st.error("Missing required configuration: path")
            return

        # Validate columns exist
        missing_cols = []
        for col in path:
            if col not in df.columns:
                missing_cols.append(col)

        if values and values not in df.columns:
            missing_cols.append(values)

        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}")
            return

        # Create chart
        try:
            fig = px.sunburst(
                df,
                path=path,
                values=values,
                title=self.title,
                height=self.height
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering sunburst chart: {str(e)}")

    def _render_custom_grid(self, df):
        """Render a grid of plots (similar to pair plot)"""
        variables = self.config.get('variables')

        if not variables:
            st.error("Missing required configuration: variables")
            return

        # Validate columns exist
        valid_vars = [var for var in variables if var in df.columns]
        if not valid_vars:
            st.error(f"No valid columns for grid. Requested: {variables}, Available: {', '.join(df.columns)}")
            return

        # Use only first few variables if too many provided
        if len(valid_vars) > 5:
            st.warning(f"Too many variables ({len(valid_vars)}). Using first 5 for better performance.")
            valid_vars = valid_vars[:5]

        # Create chart
        try:
            fig = px.scatter_matrix(
                df,
                dimensions=valid_vars,
                title=self.title,
                height=self.height
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering grid plot: {str(e)}")

    def _render_custom_time(self, df):
        """Render a time series plot with advanced features"""
        date_col = self.config.get('date_column')
        value_col = self.config.get('value_column')
        group_col = self.config.get('group_column')

        if not date_col or not value_col:
            st.error("Missing required configuration: date_column and value_column")
            return

        # Validate columns exist
        missing_cols = []
        for col in [date_col, value_col]:
            if col not in df.columns:
                missing_cols.append(col)

        if group_col and group_col not in df.columns:
            missing_cols.append(group_col)

        if missing_cols:
            st.error(f"Columns not found: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}")
            return

        # Create chart
        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                try:
                    temp_df = df.copy()
                    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
                    if temp_df[date_col].isna().sum() > 0.5 * len(temp_df):
                        st.error(f"Could not convert '{date_col}' to datetime. Too many conversion errors.")
                        return
                    df = temp_df
                except Exception as e:
                    st.error(f"Error converting '{date_col}' to datetime: {str(e)}")
                    return

            # Sort data by date
            df = df.sort_values(by=date_col)

            if group_col:
                fig = px.line(
                    df,
                    x=date_col,
                    y=value_col,
                    color=group_col,
                    title=self.title,
                    height=self.height
                )
            else:
                fig = px.line(
                    df,
                    x=date_col,
                    y=value_col,
                    title=self.title,
                    height=self.height
                )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering time series plot: {str(e)}")

    def _render_custom_summary(self, df):
        """Render a summary statistics display"""
        columns = self.config.get('columns')

        if not columns:
            st.error("Missing required configuration: columns")
            return

        # Validate columns exist
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            st.error(f"No valid columns for summary. Requested: {columns}, Available: {', '.join(df.columns)}")
            return

        # Create summary table
        try:
            # Calculate statistics
            stats_df = df[valid_cols].describe().T

            # Add additional statistics if numeric
            numeric_cols = [col for col in valid_cols if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                skew = df[numeric_cols].skew()
                kurt = df[numeric_cols].kurtosis()
                missing = df[numeric_cols].isna().sum()
                missing_pct = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)

                for col in numeric_cols:
                    if col in skew.index:
                        stats_df.loc[col, 'skew'] = skew[col]
                    if col in kurt.index:
                        stats_df.loc[col, 'kurtosis'] = kurt[col]
                    if col in missing.index:
                        stats_df.loc[col, 'missing'] = missing[col]
                        stats_df.loc[col, 'missing_pct'] = missing_pct[col]

            # Display the statistics
            st.dataframe(stats_df, use_container_width=True, height=self.height)

            # Create a visual summary if appropriate
            if len(numeric_cols) > 0:
                fig = px.bar(
                    x=stats_df.index,
                    y=stats_df['mean'],
                    error_y=stats_df['std'],
                    title="Mean Values with Standard Deviation",
                    labels={'x': 'Variable', 'y': 'Mean Value'}
                )

                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering summary statistics: {str(e)}")

    def _render_bar_chart(self, df):
        """Render a bar chart component"""
        if not self.config.get('x') or not self.config.get('y'):
            st.error("Missing required configuration: x and y")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_bar(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering bar chart: {str(e)}")

    def _render_line_chart(self, df):
        """Render a line chart component"""
        if not self.config.get('x') or not self.config.get('y'):
            st.error("Missing required configuration: x and y")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_line(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering line chart: {str(e)}")

    def _render_scatter_chart(self, df):
        """Render a scatter plot component"""
        if not self.config.get('x') or not self.config.get('y'):
            st.error("Missing required configuration: x and y")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_scatter(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering scatter plot: {str(e)}")

    def _render_pie_chart(self, df):
        """Render a pie chart component"""
        if not self.config.get('values') or not self.config.get('names'):
            st.error("Missing required configuration: values and names")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_pie(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering pie chart: {str(e)}")

    def _render_histogram(self, df):
        """Render a histogram component"""
        if not self.config.get('x'):
            st.error("Missing required configuration: x")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_histogram(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering histogram: {str(e)}")

    def _render_heatmap(self, df):
        """Render a heatmap component"""
        if not self.config.get('x') or not self.config.get('y') or not self.config.get('values'):
            st.error("Missing required configuration: x, y, and values")
            return
        try:
            cfg = {**self.config, 'title': self.title, 'height': self.height}
            st.plotly_chart(charts.build_heatmap(df, cfg), use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering heatmap: {str(e)}")

    def _render_table(self, df):
        """Render a table component"""
        columns = self.config.get('columns', None)
        rows = self.config.get('max_rows', 10)

        # Filter columns if specified
        if columns:
            display_df = df[columns].head(rows)
        else:
            display_df = df.head(rows)

        st.dataframe(display_df, use_container_width=True, height=self.height)

    def _render_metric(self, df):
        """Render a metric component"""
        column = self.config.get('column')
        calculation = self.config.get('calculation', 'mean')
        label = self.config.get('label', column)

        if not column:
            st.error("Missing required configuration: column")
            return

        # Calculate the metric
        if calculation == 'mean':
            value = df[column].mean()
        elif calculation == 'sum':
            value = df[column].sum()
        elif calculation == 'count':
            value = df[column].count()
        elif calculation == 'min':
            value = df[column].min()
        elif calculation == 'max':
            value = df[column].max()
        else:
            st.error(f"Unsupported calculation: {calculation}")
            return

        # Format value based on type
        if isinstance(value, (int, np.integer)):
            formatted_value = f"{value:,}"
        elif isinstance(value, (float, np.floating)):
            formatted_value = f"{value:,.2f}"
        else:
            formatted_value = str(value)

        # Render metric
        st.metric(label=label, value=formatted_value)


class Dashboard:
    """Class representing a dashboard with multiple components"""

    def __init__(self, name="Dashboard", description=None):
        """
        Initialize a dashboard

        Args:
            name: Name of the dashboard
            description: Optional description
        """
        self.name = name
        self.description = description
        self.components = []
        self.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_component(self, component):
        """
        Add a component to the dashboard

        Args:
            component: DashboardComponent object
        """
        self.components.append(component)
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def remove_component(self, component_id):
        """
        Remove a component from the dashboard

        Args:
            component_id: ID of the component to remove
        """
        self.components = [c for c in self.components if c.component_id != component_id]
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_json(self):
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str):
        """Create a Dashboard from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)



    def get_component(self, component_id):
        """
        Get a component by ID

        Args:
            component_id: ID of the component

        Returns:
            DashboardComponent object or None
        """
        for component in self.components:
            if component.component_id == component_id:
                return component
        return None

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'creation_date': self.creation_date,
            'last_modified': self.last_modified,
            'components': [component.to_dict() for component in self.components]
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Dashboard from a dictionary"""
        dashboard = cls(
            name=data['name'],
            description=data.get('description')
        )
        dashboard.creation_date = data.get('creation_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        dashboard.last_modified = data.get('last_modified', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        for component_data in data.get('components', []):
            dashboard.add_component(DashboardComponent.from_dict(component_data))

        return dashboard

    def render(self, df):
        """
        Render the dashboard with data

        Args:
            df: pandas DataFrame
        """
        st.title(self.name)

        if self.description:
            st.write(self.description)

        # Add metadata about the dashboard
        st.markdown(
            f"<small>Created: {self.creation_date} | Last modified: {self.last_modified} | Components: {len(self.components)}</small>",
            unsafe_allow_html=True)

        if not self.components:
            st.info("This dashboard has no components yet. Add some components to get started.")
            return

        # Down-sample large datasets so dashboard charts stay responsive
        # (mirrors the visualization page's safeguard).
        if len(df) > MAX_DASHBOARD_ROWS:
            st.caption(f"Showing a {MAX_DASHBOARD_ROWS:,}-row sample of {len(df):,} rows.")
            df = df.sample(MAX_DASHBOARD_ROWS, random_state=42)

        # Organize components into rows
        i = 0
        while i < len(self.components):
            # Get components for this row
            row_components = []
            row_width = 0

            while i < len(self.components) and row_width + self.components[i].width <= 12:
                row_components.append(self.components[i])
                row_width += self.components[i].width
                i += 1

            # Create columns for this row
            columns = st.columns([component.width for component in row_components])

            # Render components in columns
            for j, component in enumerate(row_components):
                with columns[j]:
                    component.render(df)


# Maximum rows rendered in a dashboard chart (mirrors the visualization page).
MAX_DASHBOARD_ROWS = 5000


def _dashboards_file():
    """Path to the on-disk dashboard store (created on demand)."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.app_data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'dashboards.json')


def save_dashboards(dashboards):
    """Persist all dashboards ({name: Dashboard}) to disk as JSON (best-effort)."""
    try:
        payload = {name: dash.to_dict() for name, dash in dashboards.items()}
        with open(_dashboards_file(), 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Could not save dashboards: {e}")


def load_dashboards():
    """Load dashboards from disk. Returns {name: Dashboard} (empty if none/invalid)."""
    try:
        path = _dashboards_file()
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return {name: Dashboard.from_dict(data) for name, data in payload.items()}
    except Exception as e:
        print(f"Could not load dashboards: {e}")
        return {}


def _apply_filters(st, df):
    """Render global dashboard filter widgets and return the filtered DataFrame."""
    with st.expander("Filters", expanded=False):
        chosen = st.multiselect("Filter by columns:", list(df.columns), key="dash_filter_cols")
        filtered = df
        for col in chosen:
            if pd.api.types.is_numeric_dtype(df[col]):
                lo, hi = float(df[col].min()), float(df[col].max())
                if lo < hi:
                    a, b = st.slider(f"{col} range:", lo, hi, (lo, hi), key=f"dash_filter_{col}")
                    filtered = filtered[(filtered[col] >= a) & (filtered[col] <= b)]
            else:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                picked = st.multiselect(f"{col} values:", opts, default=opts, key=f"dash_filter_{col}")
                filtered = filtered[filtered[col].astype(str).isin(picked)]
        return filtered


def export_dashboard_html(dashboard, df):
    """Combine a dashboard's charts into one self-contained HTML page (string),
    or None if no chart could be exported.

    Open the file and use the browser's "Print -> Save as PDF" to get a PDF.
    This needs no native/static-image engine (kaleido), so it can't hang.
    """
    import html as _html

    sections = []
    for comp in dashboard.components:
        fig = comp.get_figure(df)
        if fig is None:
            continue
        # Load plotly.js once (with the first chart), then reuse it.
        include_js = 'cdn' if not sections else False
        sections.append(fig.to_html(full_html=False, include_plotlyjs=include_js))

    if not sections:
        return None

    title = _html.escape(dashboard.name)
    body = "\n".join(f'<div style="margin-bottom:24px;">{s}</div>' for s in sections)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>"
    )


def dashboard_manager_ui(st, df):
    """
    Streamlit UI for managing dashboards

    Args:
        st: Streamlit instance
        df: pandas DataFrame
    """
    st.subheader("Dashboard Manager")
    st.caption("Dashboards are saved on the server. On a hosted demo they may reset when the "
               "app restarts — use Import/Export to keep a durable copy.")

    # Initialize session state for dashboards (loading any persisted ones).
    if 'dashboards' not in st.session_state:
        st.session_state.dashboards = load_dashboards()

    if 'current_dashboard' not in st.session_state:
        st.session_state.current_dashboard = None

    if 'editing_component' not in st.session_state:
        st.session_state.editing_component = None

    # Initialize delete confirmation state
    if 'delete_confirmation' not in st.session_state:
        st.session_state.delete_confirmation = False

    if 'dashboard_to_delete' not in st.session_state:
        st.session_state.dashboard_to_delete = None

    # Tabs for Dashboard Management
    create_tab, edit_tab, view_tab, import_export_tab = st.tabs([
        "Create Dashboard", "Edit Dashboard", "View Dashboard", "Import/Export"
    ])

    with create_tab:
        st.header("Create New Dashboard")

        new_dashboard_name = st.text_input("Dashboard Name:", "My Dashboard")
        new_dashboard_desc = st.text_area("Description (optional):", "")

        if st.button("Create Dashboard", key="create_dashboard_btn"):
            if new_dashboard_name in st.session_state.dashboards:
                st.warning(f"Dashboard '{new_dashboard_name}' already exists.")
            else:
                st.session_state.dashboards[new_dashboard_name] = Dashboard(
                    new_dashboard_name, new_dashboard_desc)
                st.session_state.current_dashboard = new_dashboard_name
                st.success(f"Created new dashboard: {new_dashboard_name}")
                st.rerun()

        # List existing dashboards
        st.subheader("Existing Dashboards")
        if st.session_state.dashboards:
            dashboard_data = []
            for name, dashboard in st.session_state.dashboards.items():
                dashboard_data.append({
                    'Name': name,
                    'Components': len(dashboard.components),
                    'Created': dashboard.creation_date
                })

            st.dataframe(pd.DataFrame(dashboard_data), use_container_width=True)
        else:
            st.info("No dashboards created yet.")

    with edit_tab:
        # Select a dashboard to edit
        if not st.session_state.dashboards:
            st.warning("No dashboards available for editing. Please create a dashboard first.")
        else:
            col1, col2 = st.columns([3, 1])

            with col1:
                selected_dashboard = st.selectbox(
                    "Select Dashboard to Edit:",
                    list(st.session_state.dashboards.keys()),
                    key="edit_dashboard_select"
                )

                if selected_dashboard:
                    st.session_state.current_dashboard = selected_dashboard

            with col2:
                # Handle dashboard deletion with proper confirmation flow
                if st.button("Delete Dashboard", key="delete_dashboard_btn"):
                    # Set the dashboard to delete and show confirmation dialog
                    st.session_state.dashboard_to_delete = selected_dashboard
                    st.session_state.delete_confirmation = True
                    st.rerun()

            # Handle delete confirmation if active
            if st.session_state.delete_confirmation and st.session_state.dashboard_to_delete:
                dashboard_name = st.session_state.dashboard_to_delete
                st.warning(f"Are you sure you want to delete dashboard '{dashboard_name}'?")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, delete it", key="confirm_delete"):
                        # Actually delete the dashboard
                        del st.session_state.dashboards[dashboard_name]

                        # Update current dashboard if needed
                        if st.session_state.current_dashboard == dashboard_name:
                            st.session_state.current_dashboard = None

                        # Reset confirmation state
                        st.session_state.delete_confirmation = False
                        st.session_state.dashboard_to_delete = None

                        st.success("Dashboard deleted.")
                        st.rerun()

                with col2:
                    if st.button("Cancel", key="cancel_delete"):
                        # Cancel deletion
                        st.session_state.delete_confirmation = False
                        st.session_state.dashboard_to_delete = None
                        st.rerun()

            # If a dashboard is selected, show component management
            if st.session_state.current_dashboard:
                dashboard = st.session_state.dashboards[st.session_state.current_dashboard]

                st.markdown("---")
                st.subheader(f"Editing Dashboard: {dashboard.name}")

                # Edit dashboard properties
                with st.expander("Edit Dashboard Properties", expanded=False):
                    new_name = st.text_input("Dashboard Name:", dashboard.name, key="edit_dash_name")
                    new_desc = st.text_area("Description:", dashboard.description or "", key="edit_dash_desc")

                    if st.button("Update Dashboard Properties", key="update_dash_props"):
                        if new_name != dashboard.name and new_name in st.session_state.dashboards:
                            st.warning(f"Dashboard name '{new_name}' already exists.")
                        else:
                            # Update properties
                            old_name = dashboard.name
                            dashboard.name = new_name
                            dashboard.description = new_desc
                            dashboard.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # If name changed, update dictionary key
                            if new_name != old_name:
                                st.session_state.dashboards[new_name] = dashboard
                                del st.session_state.dashboards[old_name]
                                st.session_state.current_dashboard = new_name

                            st.success("Dashboard properties updated.")
                            st.rerun()

                # Component Management Section
                st.subheader("Component Management")

                # Display current components
                if dashboard.components:
                    component_data = []
                    for i, component in enumerate(dashboard.components):
                        component_data.append({
                            'ID': component.component_id,
                            'Title': component.title,
                            'Type': component.chart_type,
                            'Width': component.width,
                            'Height': component.height,
                            'Source': component.source
                        })

                    # Show components in a data editor
                    edited_df = st.data_editor(
                        pd.DataFrame(component_data),
                        hide_index=True,
                        use_container_width=True,
                        disabled=["ID", "Type", "Source"],
                        column_config={
                            "ID": st.column_config.TextColumn("ID", width="small"),
                            "Title": st.column_config.TextColumn("Title", width="medium"),
                            "Type": st.column_config.TextColumn("Type", width="small"),
                            "Width": st.column_config.NumberColumn("Width", min_value=1, max_value=12, width="small"),
                            "Height": st.column_config.NumberColumn("Height", min_value=200, max_value=800,
                                                                    width="small"),
                            "Source": st.column_config.TextColumn("Source", width="small")
                        }
                    )

                    # Apply edits to components
                    for i, row in edited_df.iterrows():
                        component = dashboard.get_component(row['ID'])
                        if component:
                            component.title = row['Title']
                            component.width = row['Width']
                            component.height = row['Height']

                    # Component actions
                    st.subheader("Component Actions")
                    col1, col2 = st.columns(2)

                    with col1:
                        component_to_edit = st.selectbox(
                            "Select component to edit:",
                            options=[c.component_id for c in dashboard.components],
                            format_func=lambda x: next((c.title for c in dashboard.components if c.component_id == x),
                                                       x),
                            key="component_to_edit"
                        )

                        if st.button("Edit Component", key="edit_component_btn"):
                            if component_to_edit:
                                st.session_state.editing_component = component_to_edit
                                st.rerun()

                    with col2:
                        component_to_remove = st.selectbox(
                            "Select component to remove:",
                            options=[c.component_id for c in dashboard.components],
                            format_func=lambda x: next((c.title for c in dashboard.components if c.component_id == x),
                                                       x),
                            key="component_to_remove"
                        )

                        if st.button("Remove Component", key="remove_component_btn"):
                            if component_to_remove:
                                dashboard.remove_component(component_to_remove)
                                st.success("Component removed.")
                                st.rerun()

                # Add new component section
                st.subheader("Add New Component")
                with st.expander("Add New Component", expanded=False):
                    component_title = st.text_input("Component Title:", "New Component", key="new_component_title")

                    chart_types = [
                        "bar", "line", "scatter", "pie", "histogram",
                        "box", "heatmap", "table", "metric"
                    ]
                    chart_type = st.selectbox("Chart Type:", chart_types, key="new_component_type")

                    # Configuration based on chart type
                    config = {}

                    if chart_type in ["bar", "line"]:
                        x_col = st.selectbox(f"X-axis Column:", df.columns, key="new_x_col")
                        y_col = st.selectbox(f"Y-axis Column:",
                                             [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                             key="new_y_col")
                        aggregation = st.selectbox("Aggregation:",
                                                   ["none", "sum", "mean", "count"],
                                                   key="new_aggregation")

                        config = {
                            'x': x_col,
                            'y': y_col,
                            'aggregation': aggregation
                        }

                        # Additional options for bar charts
                        if chart_type == "bar":
                            orientation = st.radio("Orientation:", ["vertical", "horizontal"], key="new_orientation")
                            config['orientation'] = orientation

                    elif chart_type == "scatter":
                        x_col = st.selectbox(f"X-axis Column:",
                                             [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                             key="new_scatter_x")
                        y_col = st.selectbox(f"Y-axis Column:",
                                             [c for c in df.columns if
                                              pd.api.types.is_numeric_dtype(df[c]) and c != x_col],
                                             key="new_scatter_y")

                        color_col = st.selectbox("Color Column (optional):",
                                                 ["None"] + [c for c in df.columns if c not in [x_col, y_col]],
                                                 key="new_scatter_color")
                        if color_col != "None":
                            config['color'] = color_col

                        config['x'] = x_col
                        config['y'] = y_col

                    elif chart_type == "pie":
                        values_col = st.selectbox(f"Values Column:",
                                                  [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                                  key="new_pie_values")
                        names_col = st.selectbox(f"Names Column:",
                                                 [c for c in df.columns if c != values_col],
                                                 key="new_pie_names")

                        config = {
                            'values': values_col,
                            'names': names_col
                        }

                    elif chart_type == "histogram":
                        x_col = st.selectbox(f"Column for Histogram:",
                                             [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                             key="new_hist_x")
                        bins = st.slider("Number of Bins:", 5, 100, 20, key="new_hist_bins")

                        config = {
                            'x': x_col,
                            'bins': bins
                        }

                    elif chart_type == "box":
                        y_col = st.selectbox(f"Value Column:",
                                             [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                             key="new_box_y")
                        x_col = st.selectbox("Category Column (optional):",
                                             ["None"] + [c for c in df.columns if c != y_col],
                                             key="new_box_x")

                        config = {'y': y_col}
                        if x_col != "None":
                            config['x'] = x_col

                    elif chart_type == "heatmap":
                        x_col = st.selectbox(f"X-axis Column:", df.columns, key="new_heatmap_x")
                        y_col = st.selectbox(f"Y-axis Column:",
                                             [c for c in df.columns if c != x_col],
                                             key="new_heatmap_y")
                        values_col = st.selectbox(f"Values Column:",
                                                  [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                                  key="new_heatmap_values")
                        aggregation = st.selectbox("Aggregation:", ["sum", "mean", "count"], key="new_heatmap_agg")

                        config = {
                            'x': x_col,
                            'y': y_col,
                            'values': values_col,
                            'aggregation': aggregation
                        }

                    elif chart_type == "table":
                        selected_columns = st.multiselect("Columns to Display:",
                                                          df.columns.tolist(),
                                                          default=df.columns[:5].tolist(),
                                                          key="new_table_cols")
                        max_rows = st.slider("Maximum Rows:", 5, 50, 10, key="new_table_rows")

                        config = {
                            'columns': selected_columns,
                            'max_rows': max_rows
                        }

                    elif chart_type == "metric":
                        metric_col = st.selectbox(f"Column for Metric:",
                                                  [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                                                  key="new_metric_col")
                        calculation = st.selectbox("Calculation:",
                                                   ["mean", "sum", "count", "min", "max"],
                                                   key="new_metric_calc")
                        label = st.text_input("Label (optional):", "", key="new_metric_label")

                        config = {
                            'column': metric_col,
                            'calculation': calculation
                        }

                        if label:
                            config['label'] = label

                    # Component layout
                    col1, col2 = st.columns(2)
                    with col1:
                        width = st.slider("Width (columns):", 1, 12, 6, key="new_comp_width")
                    with col2:
                        height = st.slider("Height (pixels):", 200, 800, 400, key="new_comp_height")

                    if st.button("Add Component", key="add_component_btn"):
                        component = DashboardComponent(
                            title=component_title,
                            chart_type=chart_type,
                            config=config,
                            width=width,
                            height=height
                        )

                        dashboard.add_component(component)
                        st.success(f"Added component: {component_title}")
                        st.rerun()

                # Component editor
                if st.session_state.editing_component:
                    component = dashboard.get_component(st.session_state.editing_component)

                    if component:
                        st.markdown("---")
                        st.subheader(f"Editing Component: {component.title}")

                        # Edit component configuration based on chart type
                        new_title = st.text_input("Component Title:", component.title, key="edit_comp_title")

                        # Create a copy of the configuration
                        new_config = dict(component.config)

                        if component.chart_type in ["bar", "line"]:
                            new_config['x'] = st.selectbox(
                                f"X-axis Column:", df.columns,
                                index=list(df.columns).index(new_config.get('x', df.columns[0])),
                                key="edit_comp_x"
                            )

                            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                            y_index = numeric_cols.index(new_config.get('y')) if new_config.get(
                                'y') in numeric_cols else 0
                            new_config['y'] = st.selectbox(
                                f"Y-axis Column:", numeric_cols,
                                index=y_index,
                                key="edit_comp_y"
                            )

                            agg_options = ["none", "sum", "mean", "count"]
                            agg_index = agg_options.index(new_config.get('aggregation', 'none'))
                            new_config['aggregation'] = st.selectbox(
                                "Aggregation:", agg_options,
                                index=agg_index,
                                key="edit_comp_agg"
                            )

                            if component.chart_type == "bar":
                                orientation_options = ["vertical", "horizontal"]
                                orientation_index = orientation_options.index(new_config.get('orientation', 'vertical'))
                                new_config['orientation'] = st.radio(
                                    "Orientation:", orientation_options,
                                    index=orientation_index,
                                    key="edit_comp_orientation"
                                )

                        # Repeat for other chart types...

                        # Component layout
                        col1, col2 = st.columns(2)
                        with col1:
                            width = st.slider("Width (columns):", 1, 12, component.width, key="edit_comp_width")
                        with col2:
                            height = st.slider("Height (pixels):", 200, 800, component.height, key="edit_comp_height")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save Changes", key="save_comp_changes"):
                                # Update component
                                component.title = new_title
                                component.config = new_config
                                component.width = width
                                component.height = height

                                # Update dashboard
                                dashboard.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                st.session_state.editing_component = None
                                st.success("Component updated.")
                                st.rerun()

                        with col2:
                            if st.button("Cancel Editing", key="cancel_edit_comp"):
                                st.session_state.editing_component = None
                                st.rerun()

    with view_tab:
        # View selected dashboard
        if not st.session_state.dashboards:
            st.warning("No dashboards available for viewing. Please create a dashboard first.")
        else:
            selected_dashboard = st.selectbox(
                "Select Dashboard to View:",
                list(st.session_state.dashboards.keys()),
                key="view_dashboard_select"
            )

            if selected_dashboard:
                dashboard = st.session_state.dashboards[selected_dashboard]

                # Global filters applied to the data before rendering.
                filtered_df = _apply_filters(st, df)

                # Export the dashboard's charts to one self-contained HTML file
                # (open it, then "Print -> Save as PDF" in the browser).
                if st.button("Export dashboard as HTML", key="export_html_btn"):
                    html_doc = export_dashboard_html(dashboard, filtered_df)
                    if html_doc:
                        st.download_button(
                            "Download dashboard HTML", data=html_doc.encode("utf-8"),
                            file_name=f"{dashboard.name}.html", mime="text/html",
                            key="download_html_btn"
                        )
                    else:
                        st.warning("No exportable charts found (custom chart types are not "
                                   "yet supported by export).")

                # Add a fullscreen mode option
                fullscreen = st.checkbox("Fullscreen Mode", key="fullscreen_view")

                st.markdown("---")

                if fullscreen:
                    dashboard.render(filtered_df)
                else:
                    st.subheader(f"Dashboard Preview: {dashboard.name}")
                    dashboard.render(filtered_df)

    with import_export_tab:
        export_tab, import_tab = st.tabs(["Export Dashboard", "Import Dashboard"])

        with export_tab:
            # Export dashboard to JSON
            if not st.session_state.dashboards:
                st.warning("No dashboards available for export. Please create a dashboard first.")
            else:
                dashboard_to_export = st.selectbox(
                    "Select Dashboard to Export:",
                    list(st.session_state.dashboards.keys()),
                    key="export_dashboard_select"
                )

                if dashboard_to_export:
                    dashboard = st.session_state.dashboards[dashboard_to_export]

                    # Create JSON for export
                    json_str = dashboard.to_json()

                    # Create download link for JSON
                    b64 = base64.b64encode(json_str.encode()).decode()
                    filename = f"{dashboard.name.lower().replace(' ', '_')}_dashboard.json"
                    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Dashboard JSON</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Show JSON
                    with st.expander("View Dashboard JSON"):
                        st.code(json_str, language="json")

                    st.caption("Tip: expand the JSON above to select and copy it, or use the download link.")

        with import_tab:
            # Import dashboard from JSON
            import_method = st.radio(
                "Import Method:",
                ["Paste JSON", "Upload File"],
                horizontal=True,
                key="import_method"
            )

            if import_method == "Paste JSON":
                json_input = st.text_area("Paste Dashboard JSON:", height=200, key="paste_json")

                if json_input:
                    try:
                        imported_dashboard = Dashboard.from_json(json_input)
                        st.success(f"Successfully parsed dashboard: {imported_dashboard.name}")

                        # Option to rename
                        with st.expander("Rename Dashboard", expanded=False):
                            new_name = st.text_input("Dashboard Name (leave empty to keep original):",
                                                     key="import_new_name")

                            if st.button("Update Name", key="update_import_name"):
                                if new_name:
                                    if new_name in st.session_state.dashboards:
                                        st.warning(
                                            f"Dashboard '{new_name}' already exists. Please choose a different name.")
                                    else:
                                        imported_dashboard.name = new_name
                                        st.success(f"Dashboard name updated to: {new_name}")

                        if st.button("Import Dashboard", key="import_dash_btn"):
                            if imported_dashboard.name in st.session_state.dashboards:
                                st.warning(
                                    f"Dashboard '{imported_dashboard.name}' already exists. Please choose a different name.")
                            else:
                                st.session_state.dashboards[imported_dashboard.name] = imported_dashboard
                                st.session_state.current_dashboard = imported_dashboard.name
                                st.success(f"Imported dashboard: {imported_dashboard.name}")
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error importing dashboard: {str(e)}")

            else:  # Upload File
                uploaded_file = st.file_uploader("Upload dashboard JSON file:", type=["json"], key="upload_json")

                if uploaded_file:
                    try:
                        json_str = uploaded_file.read().decode("utf-8")
                        imported_dashboard = Dashboard.from_json(json_str)
                        st.success(f"Successfully parsed dashboard from file: {imported_dashboard.name}")

                        # Option to rename
                        with st.expander("Rename Dashboard", expanded=False):
                            new_name = st.text_input("Dashboard Name (leave empty to keep original):",
                                                     key="import_file_new_name")

                            if st.button("Update Name", key="update_import_file_name"):
                                if new_name:
                                    if new_name in st.session_state.dashboards:
                                        st.warning(
                                            f"Dashboard '{new_name}' already exists. Please choose a different name.")
                                    else:
                                        imported_dashboard.name = new_name
                                        st.success(f"Dashboard name updated to: {new_name}")

                        if st.button("Import Dashboard from File", key="import_file_dash_btn"):
                            if imported_dashboard.name in st.session_state.dashboards:
                                st.warning(
                                    f"Dashboard '{imported_dashboard.name}' already exists. Please choose a different name.")
                            else:
                                st.session_state.dashboards[imported_dashboard.name] = imported_dashboard
                                st.session_state.current_dashboard = imported_dashboard.name
                                st.success(f"Imported dashboard: {imported_dashboard.name}")
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error importing dashboard from file: {str(e)}")

    # Persist all dashboards to disk so they survive a refresh / restart.
    save_dashboards(st.session_state.dashboards)