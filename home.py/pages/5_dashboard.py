import streamlit as st
import pandas as pd
from utils.dashboard_module import dashboard_manager_ui
from config import Config  # Import Config for theme

# Set page configuration for better appearance
st.set_page_config(
    page_title="Interactive Dashboards",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply the CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)

# Page header with back navigation
st.markdown('<div class="page-header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
    if st.button('← ', key='back_button', help="Return to visualization page"):
        st.switch_page("pages/4_visualization.py")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="margin: 0; color: white;">Interactive Dashboards</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Check for data
if 'data' not in st.session_state or st.session_state.data is None:
    st.markdown('<div class="card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
    st.warning("No data loaded. Please upload a dataset first.")
    if st.button('Go to Data Upload', use_container_width=True):
        st.switch_page("pages/1_upload_data.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Main dashboard content card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dashboard Manager</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Create, manage, and share interactive dashboards from your data</div>', unsafe_allow_html=True)

# Call the dashboard manager UI
dashboard_manager_ui(st, st.session_state.data)

st.markdown('</div>', unsafe_allow_html=True)

# Navigation at the bottom
st.markdown('<div style="display: flex; justify-content: space-between; margin-top: 2rem;">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button('← Previous: Visualization', key='prev_button', use_container_width=True):
        st.switch_page("pages/4_visualization.py")
with col2:
    st.button('Next: Export Data →', key='next_button', disabled=True, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer with help information
st.markdown('<div style="text-align: center; padding: 10px; color: #888; margin-top: 20px;">', unsafe_allow_html=True)
st.markdown('<small>Use the navigation buttons to move between pages. You can add visualizations directly from the visualization page.</small>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)