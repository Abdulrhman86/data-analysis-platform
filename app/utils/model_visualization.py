# utils/model_visualization.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def plot_confusion_matrix(cm, classes=None, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix using plotly

    Args:
        cm: confusion matrix array
        classes: list of class names
        normalize: if True, normalize confusion matrix
        title: plot title

    Returns:
        plotly figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # If classes are not provided, use numeric labels
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    # Create annotation text
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append({
                'x': j,
                'y': i,
                'text': format(cm[i, j], fmt),
                'showarrow': False,
                'font': {'color': 'white' if cm[i, j] > cm.max() / 2 else 'black'}
            })

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale='Blues',
        hoverongaps=False))

    # Add annotations
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label'),
        annotations=annotations,
        width=600,
        height=600
    )

    return fig


def plot_roc_curve(roc_data, title='ROC Curve'):
    """
    Plot ROC curve using plotly

    Args:
        roc_data: dict containing fpr, tpr, and auc
        title: plot title

    Returns:
        plotly figure
    """
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=roc_data['fpr'],
        y=roc_data['tpr'],
        mode='lines',
        name=f'ROC (AUC = {roc_data["auc"]:.3f})'
    ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Chance'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        width=600,
        height=500,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )

    return fig


def plot_precision_recall_curve(pr_data, title='Precision-Recall Curve'):
    """
    Plot precision-recall curve using plotly

    Args:
        pr_data: dict containing precision, recall, and avg_precision
        title: plot title

    Returns:
        plotly figure
    """
    fig = go.Figure()

    # Add precision-recall curve
    fig.add_trace(go.Scatter(
        x=pr_data['recall'],
        y=pr_data['precision'],
        mode='lines',
        name=f'PR (Avg Precision = {pr_data["avg_precision"]:.3f})'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Recall'),
        yaxis=dict(title='Precision'),
        width=600,
        height=500,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )

    return fig


def plot_learning_curve(learning_curve_data, title='Learning Curve'):
    """
    Plot learning curve using plotly

    Args:
        learning_curve_data: dict containing train_sizes, train_scores, and test_scores
        title: plot title

    Returns:
        plotly figure
    """
    train_sizes = learning_curve_data['train_sizes']
    train_scores_mean = np.mean(learning_curve_data['train_scores'], axis=1)
    train_scores_std = np.std(learning_curve_data['train_scores'], axis=1)
    test_scores_mean = np.mean(learning_curve_data['test_scores'], axis=1)
    test_scores_std = np.std(learning_curve_data['test_scores'], axis=1)

    fig = go.Figure()

    # Add training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores_mean,
        mode='lines+markers',
        name='Training score',
        line=dict(color='blue')
    ))

    # Add training score error bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_scores_mean + train_scores_std, (train_scores_mean - train_scores_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False
    ))

    # Add test scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_scores_mean,
        mode='lines+markers',
        name='Cross-validation score',
        line=dict(color='red')
    ))

    # Add test score error bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([test_scores_mean + test_scores_std, (test_scores_mean - test_scores_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Training examples'),
        yaxis=dict(title='Score'),
        width=700,
        height=500
    )

    return fig


def plot_feature_importance(feature_importance, top_n=None, title='Feature Importance'):
    """
    Plot feature importance using plotly

    Args:
        feature_importance: pandas Series with feature names as index and importance as values
        top_n: number of top features to show (if None, show all)
        title: plot title

    Returns:
        plotly figure
    """
    if feature_importance is None:
        return None

    # Sort and get top N features if specified
    sorted_importance = feature_importance.sort_values(ascending=True)
    if top_n is not None and top_n < len(sorted_importance):
        sorted_importance = sorted_importance.tail(top_n)

    fig = px.bar(
        x=sorted_importance.values,
        y=sorted_importance.index,
        orientation='h',
        title=title
    )

    # Update layout
    fig.update_layout(
        xaxis=dict(title='Importance'),
        yaxis=dict(title='Feature'),
        width=700,
        height=max(400, 20 * len(sorted_importance))
    )

    return fig


def plot_residuals(residuals, y_pred, title='Residuals Plot'):
    """
    Plot residuals for regression models

    Args:
        residuals: array of residuals
        y_pred: array of predicted values
        title: plot title

    Returns:
        plotly figure
    """
    fig = go.Figure()

    # Add scatter plot of residuals
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='blue', opacity=0.5),
        name='Residuals'
    ))

    # Add horizontal line at y=0
    fig.add_trace(go.Scatter(
        x=[min(y_pred), max(y_pred)],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Zero Line'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted Values'),
        yaxis=dict(title='Residuals'),
        width=700,
        height=500,
        showlegend=False
    )

    return fig


def plot_regression_results(y_test, y_pred, title='Actual vs Predicted'):
    """
    Plot actual vs predicted values for regression models

    Args:
        y_test: array of actual values
        y_pred: array of predicted values
        title: plot title

    Returns:
        plotly figure
    """
    fig = go.Figure()

    # Add scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', opacity=0.5),
        name='Predictions'
    ))

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Actual Values'),
        yaxis=dict(title='Predicted Values'),
        width=700,
        height=500
    )

    return fig


def plot_model_comparison(models_results, metric_name, higher_is_better=True):
    """
    Create a comparison bar chart for multiple models

    Args:
        models_results: dict mapping model names to their metrics dictionaries
        metric_name: name of the metric to compare
        higher_is_better: if True, higher metric values are better

    Returns:
        plotly figure
    """
    model_names = list(models_results.keys())
    metric_values = [models_results[model].get(metric_name, 0) for model in model_names]

    # Sort by metric value
    if higher_is_better:
        sorted_indices = np.argsort(metric_values)
    else:
        sorted_indices = np.argsort(metric_values)[::-1]

    sorted_models = [model_names[i] for i in sorted_indices]
    sorted_values = [metric_values[i] for i in sorted_indices]

    # Choose color based on whether higher is better
    colors = ['#4CAF50' if higher_is_better else '#F44336' for _ in sorted_models]

    fig = go.Figure([
        go.Bar(
            x=sorted_models,
            y=sorted_values,
            marker_color=colors
        )
    ])

    # Update layout
    fig.update_layout(
        title=f'Model Comparison: {metric_name}',
        xaxis=dict(title='Model'),
        yaxis=dict(title=metric_name),
        width=700,
        height=500
    )

    return fig