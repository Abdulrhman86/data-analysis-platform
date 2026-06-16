import streamlit as st
import pandas as pd
import os
import html
from utils.preprocessing_utils import (
    read_csv_robust,
    prepare_excel_datetime_columns,
    finalize_dataframe,
)
from config import Config, Paths

# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📊",
    page_title="Data Upload"
)

# Apply CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)
st.markdown(Config.stepper(1), unsafe_allow_html=True)

def _validate_and_finalize(df):
    """Guard empty files (UI layer), then run the shared finalize cleanup
    (column de-dup, numeric-string coercion, date detection)."""
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        st.error("The file appears to be empty (no rows or no columns). "
                 "Please upload a non-empty CSV or Excel file.")
        return None
    return finalize_dataframe(df)


# Function to read and process uploaded files
def read_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            # Add options for CSV files
            with st.expander("CSV Import Options", expanded=False):
                header_option = st.checkbox("First row is header", value=True)
                header = 0 if header_option else None

            df = read_csv_robust(uploaded_file, header)
            return _validate_and_finalize(df)

        elif file_extension in ['xlsx', 'xls']:
            # Add options for Excel files
            with st.expander("Excel Import Options", expanded=False):
                try:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheets = excel_file.sheet_names
                    sheet_name = st.selectbox("Select sheet", sheets, index=0)
                    header_option = st.checkbox("First row is header", value=True)
                    header = 0 if header_option else None

                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)
                    df = prepare_excel_datetime_columns(df)
                    return _validate_and_finalize(df)
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
                    return None
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# Initialize session state variables
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'file' not in st.session_state:
    st.session_state.file = ""
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header with back navigation
st.markdown('<div class="page-header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
    if st.button('← ', key='back_button', help="Return to home page"):
        st.switch_page("home.py")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="margin: 0; color: white;">Upload Dataset</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# File upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Select Data Source</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=Config.ALLOWED_EXTENSIONS)
    st.caption("🔒 This is a public demo — please don't upload confidential or personal data.")

    if st.button("✨ Try it with sample data", help="Load a sample sales dataset to explore the app"):
        try:
            sample_df = pd.read_csv(os.path.join(Paths.STATIC_DIR, "sample_sales.csv"))
            sample_df = _validate_and_finalize(sample_df)
            if sample_df is not None:
                st.session_state.data = sample_df
                st.session_state.file = "sample_sales.csv"
                st.session_state.file_name = "sample_sales.csv"
                st.rerun()
        except Exception as e:
            st.error(f"Could not load sample data: {e}")

    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > Config.MAX_FILE_SIZE_MB:
            st.error(f"File is {size_mb:.1f} MB — the limit is {Config.MAX_FILE_SIZE_MB} MB. "
                     "Please upload a smaller file or a sample of your data.")
            st.stop()

        st.session_state.file = uploaded_file
        st.session_state.file_name = uploaded_file.name
        st.session_state.data = read_file(uploaded_file)

        if st.session_state.data is not None:
            st.success(f"File '{uploaded_file.name}' successfully uploaded!")

            # Display file size
            file_size_kb = round(uploaded_file.size / 1024, 2)
            if file_size_kb > 1024:
                file_size_mb = round(file_size_kb / 1024, 2)
                st.info(f"File size: {file_size_mb} MB")
            else:
                st.info(f"File size: {file_size_kb} KB")

with col2:
    st.markdown('<p class="metric-label">Currently Selected</p>', unsafe_allow_html=True)
    if st.session_state.file_name:
        st.markdown(f'<p class="metric-value">{html.escape(st.session_state.file_name)}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #6B7280;">No file selected</p>', unsafe_allow_html=True)

    # Feature for future implementation
    if st.button('Recent Files', disabled=True, help="Coming soon: Access recently used datasets"):
        pass  # Placeholder for future functionality

st.markdown('</div>', unsafe_allow_html=True)

# Display data preview if available
if st.session_state.data is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)

    st.dataframe(st.session_state.data.head(10), use_container_width=True)

    # Dataset information
    with st.expander("Dataset Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            num_rows, num_columns = st.session_state.data.shape
            missing_values = st.session_state.data.isnull().sum().sum()
            categorical_columns = st.session_state.data.select_dtypes(include=['object', 'category']).columns
            numerical_columns = st.session_state.data.select_dtypes(include=['number']).columns

            st.markdown(Config.metric_card("Rows", f"{num_rows:,}"), unsafe_allow_html=True)
            st.markdown(Config.metric_card("Columns", f"{num_columns}"), unsafe_allow_html=True)
            st.markdown(Config.metric_card("Missing Values", f"{missing_values:,}"), unsafe_allow_html=True)

        with col2:
            st.markdown(Config.metric_card("Categorical Features", f"{len(categorical_columns)}"), unsafe_allow_html=True)
            st.markdown(Config.metric_card("Numerical Features", f"{len(numerical_columns)}"), unsafe_allow_html=True)
            quality_score = 100 - min(100, (missing_values / (num_rows * num_columns) * 100 if num_rows * num_columns > 0 else 0))
            st.markdown(Config.metric_card("Data completeness", f"{quality_score:.2f}%"), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation buttons
    cols = st.columns([3, 3, 1])
    with cols[2]:
        if st.button('Continue →', help="Proceed to data quality"):
            if st.session_state.data is not None:
                st.switch_page("pages/2_data_quality.py")