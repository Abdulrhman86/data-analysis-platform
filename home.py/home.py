import streamlit as st
import os
from config import Config, Paths

# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title=Config.APP_NAME
)

# Apply the global CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)

# Header section
st.markdown('<div class="center-content">', unsafe_allow_html=True)
st.markdown(f'<h1 class="title">{Config.APP_NAME}</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your all-in-one solution for data preprocessing, visualization, and insights</p>',
            unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main content in two columns
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white;">Here to take care of your data</h3>', unsafe_allow_html=True)
    st.markdown('<p>Our platform offers a complete workflow:</p>', unsafe_allow_html=True)

    # Checkboxes with green check marks
    st.markdown("""
    <div class="checkbox-item">
        <span class="green-check">✅</span> Easy data upload and inspection
    </div>
    <div class="checkbox-item">
        <span class="green-check">✅</span> Powerful preprocessing capabilities
    </div>
    <div class="checkbox-item">
        <span class="green-check">✅</span> Interactive visualizations
    </div>
    <div class="checkbox-item">
        <span class="green-check">✅</span> Custom dashboard creation
    </div>
    <p style="margin-top: 1rem;">Get started now to transform your data journey!</p>
    """, unsafe_allow_html=True)

    # Button
    if st.button('Get Started'):
        st.switch_page("pages/1_upload_data.py")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(os.path.join(Paths.STATIC_DIR, 'laptop.PNG'))
    st.markdown('</div>', unsafe_allow_html=True)

# Add a simple footer
st.markdown("""
<div style="text-align: center; margin-top: 1rem; color: #6B7280; font-size: 0.8rem;">
    © 2025 Data Analysis Platform • All rights reserved
</div>
""", unsafe_allow_html=True)