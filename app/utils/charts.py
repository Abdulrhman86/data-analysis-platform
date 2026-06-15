"""Shared chart-building core.

Pure functions that turn a ``(DataFrame, config)`` pair into a Plotly figure.
The dashboard engine renders through these so charts are built one way only
(no duplicated bar/line/scatter/... implementations). ``config`` may include a
``title`` and ``height`` alongside the chart-specific keys.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _aggregate(df, x, y, how):
    """Group ``df`` by ``x`` and aggregate ``y``. ``how`` in sum/mean/count/none."""
    if how in (None, "none"):
        return df
    if how in ("sum", "mean", "count"):
        return getattr(df.groupby(x)[y], how)().reset_index()
    return df


def build_bar(df, config):
    x, y = config.get("x"), config.get("y")
    data = _aggregate(df, x, y, config.get("aggregation", "sum"))
    if config.get("orientation", "vertical") == "vertical":
        return px.bar(data, x=x, y=y, color=config.get("color"),
                      title=config.get("title"), height=config.get("height"))
    return px.bar(data, x=y, y=x, color=config.get("color"), orientation="h",
                  title=config.get("title"), height=config.get("height"))


def build_line(df, config):
    x, y = config.get("x"), config.get("y")
    data = _aggregate(df, x, y, config.get("aggregation", "none"))
    return px.line(data, x=x, y=y, color=config.get("color"),
                   title=config.get("title"), height=config.get("height"))


def build_scatter(df, config):
    return px.scatter(df, x=config.get("x"), y=config.get("y"),
                      color=config.get("color"), size=config.get("size"),
                      title=config.get("title"), height=config.get("height"))


def build_pie(df, config):
    return px.pie(df, values=config.get("values"), names=config.get("names"),
                  title=config.get("title"), height=config.get("height"))


def build_histogram(df, config):
    return px.histogram(df, x=config.get("x"), color=config.get("color"),
                        nbins=config.get("bins", 20), opacity=config.get("opacity"),
                        title=config.get("title"), height=config.get("height"))


def build_box(df, config):
    return px.box(df, y=config.get("y"), x=config.get("x"),
                  title=config.get("title"), height=config.get("height"))


def build_violin(df, config):
    return px.violin(df, y=config.get("y"), x=config.get("x"), box=True, points="all",
                     title=config.get("title"), height=config.get("height"))


def build_kde(df, config):
    """Density histogram overlaid with a real Gaussian-KDE curve.

    (The previous dashboard KDE added ``density_contour(...).data[0]``, a
    degenerate 2-D contour trace, so no actual KDE was drawn.)
    """
    x = config.get("x")
    fig = px.histogram(df, x=x, histnorm="probability density",
                       title=config.get("title"), height=config.get("height"))
    fig.update_layout(bargap=0.05)
    series = pd.to_numeric(df[x], errors="coerce").dropna()
    if len(series) > 1 and series.nunique() > 1:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series)
            xs = np.linspace(series.min(), series.max(), 200)
            fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", name="KDE"))
        except Exception:
            pass  # KDE overlay is best-effort; the histogram still renders.
    return fig


def build_heatmap(df, config):
    x, y, values = config.get("x"), config.get("y"), config.get("values")
    how = config.get("aggregation", "sum")
    aggfunc = how if how in ("sum", "mean", "count") else "sum"
    pivot = df.pivot_table(index=y, columns=x, values=values, aggfunc=aggfunc)
    return px.imshow(pivot, title=config.get("title"), height=config.get("height"))


# chart_type -> builder. Shared by the dashboard engine.
BUILDERS = {
    "bar": build_bar,
    "line": build_line,
    "scatter": build_scatter,
    "pie": build_pie,
    "histogram": build_histogram,
    "box": build_box,
    "violin": build_violin,
    "kde": build_kde,
    "heatmap": build_heatmap,
}
