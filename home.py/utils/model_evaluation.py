# utils/model_evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve


def get_classification_report(model, X_test, y_test, as_dataframe=True):
    """
    Generate classification report

    Args:
        model: trained model
        X_test: test features
        y_test: test target
        as_dataframe: if True, return as pandas DataFrame

    Returns:
        classification report as DataFrame or string
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=as_dataframe)

    if as_dataframe:
        return pd.DataFrame(report).transpose()
    return report


def get_confusion_matrix(model, X_test, y_test, normalize=False):
    """
    Generate confusion matrix

    Args:
        model: trained model
        X_test: test features
        y_test: test target
        normalize: if True, normalize confusion matrix

    Returns:
        confusion matrix array
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


def get_roc_curve_data(model, X_test, y_test):
    """
    Get ROC curve data for binary classification

    Args:
        model: trained model
        X_test: test features
        y_test: test target

    Returns:
        dict: containing fpr, tpr, thresholds, and auc
    """
    # Check if binary classification
    classes = np.unique(y_test)
    if len(classes) != 2:
        raise ValueError("ROC curve is only defined for binary classification")

    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc
    }


def get_precision_recall_curve_data(model, X_test, y_test):
    """
    Get precision-recall curve data for binary classification

    Args:
        model: trained model
        X_test: test features
        y_test: test target

    Returns:
        dict: containing precision, recall, thresholds, and average precision
    """
    # Check if binary classification
    classes = np.unique(y_test)
    if len(classes) != 2:
        raise ValueError("Precision-Recall curve is only defined for binary classification")

    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'avg_precision': avg_precision
    }


def get_learning_curve_data(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Get learning curve data

    Args:
        model: trained model
        X: features
        y: target
        cv: number of cross-validation folds
        train_sizes: array of training size proportions

    Returns:
        dict: containing train_sizes, train_scores, and test_scores
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes
    )

    return {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores
    }


def get_residuals(model, X, y):
    """
    Calculate residuals for regression models

    Args:
        model: trained model
        X: features
        y: target

    Returns:
        numpy array: residuals
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    return residuals