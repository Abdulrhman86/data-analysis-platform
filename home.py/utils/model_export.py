# utils/model_export.py
import pickle
import os
import base64
import zipfile
import io
import json
import numpy as np


def _json_default(obj):
    """Fallback serializer so numpy types/arrays in metadata are JSON-safe."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def save_model(model, model_name, directory='models'):
    """
    Save a trained model to disk

    Args:
        model: trained model
        model_name: name of the model
        directory: directory to save the model

    Returns:
        str: path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Create safe filename
    filename = os.path.join(directory, f"{model_name.replace(' ', '_').lower()}.pkl")

    # Save model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    return filename


def load_model(filepath):
    """
    Load a model from disk

    Args:
        filepath: path to the saved model

    Returns:
        trained model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    return model


def create_download_link(model, model_name, metadata=None):
    """
    Create a download link for a trained model

    Args:
        model: trained model
        model_name: name of the model
        metadata: optional dictionary with additional information

    Returns:
        str: HTML download link
    """
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save model
        model_buffer = io.BytesIO()
        pickle.dump(model, model_buffer)
        model_buffer.seek(0)
        zf.writestr('model.pkl', model_buffer.read())

        # Save metadata if provided
        if metadata:
            zf.writestr('metadata.json', json.dumps(metadata, indent=2, default=_json_default))

    # Create download link
    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    filename = f"{model_name.replace(' ', '_').lower()}_model.zip"
    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Download {model_name} Model</a>'

    return href


def create_streamlit_download_button(model, model_name, metadata=None):
    """
    Create a Streamlit download button for a trained model

    Args:
        model: trained model
        model_name: name of the model
        metadata: optional dictionary with additional information

    Returns:
        None (displays a download button)
    """
    import streamlit as st

    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save model
        model_buffer = io.BytesIO()
        pickle.dump(model, model_buffer)
        model_buffer.seek(0)
        zf.writestr('model.pkl', model_buffer.read())

        # Save metadata if provided
        if metadata:
            zf.writestr('metadata.json', json.dumps(metadata, indent=2, default=_json_default))

    # Create download button
    zip_buffer.seek(0)
    filename = f"{model_name.replace(' ', '_').lower()}_model.zip"

    st.download_button(
        label=f"Download {model_name} Model",
        data=zip_buffer,
        file_name=filename,
        mime="application/zip"
    )


def export_full_pipeline(models, data_info, preprocessing_steps=None):
    """
    Export the full machine learning pipeline including models and preprocessing steps

    Args:
        models: dict of trained models
        data_info: information about the dataset
        preprocessing_steps: list of preprocessing steps

    Returns:
        io.BytesIO: buffer containing the zipped pipeline
    """
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save models
        for model_name, model in models.items():
            model_buffer = io.BytesIO()
            pickle.dump(model, model_buffer)
            model_buffer.seek(0)
            zf.writestr(f'models/{model_name.replace(" ", "_").lower()}.pkl', model_buffer.read())

        # Save data info
        zf.writestr('data_info.json', json.dumps(data_info, indent=2, default=_json_default))

        # Save preprocessing steps if provided
        if preprocessing_steps:
            zf.writestr('preprocessing_steps.json', json.dumps(preprocessing_steps, indent=2, default=_json_default))

    zip_buffer.seek(0)
    return zip_buffer