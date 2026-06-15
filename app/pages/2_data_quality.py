import streamlit as st
import pandas as pd
import html
from utils.data_quality import enhanced_data_quality
from config import Config

# Configure page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Data Quality Assessment"
)

# Apply the CSS from config
st.markdown(Config.get_css(), unsafe_allow_html=True)
st.markdown(Config.stepper(2), unsafe_allow_html=True)

# Additional CSS for search feature
st.markdown("""
<style>
    .search-container {
        margin-bottom: 16px;
    }
    .no-results {
        padding: 16px;
        background-color: rgba(59, 130, 246, 0.1);
        border-radius: 6px;
        margin-top: 12px;
    }
    .search-results-count {
        color: #9CA3AF;
        font-size: 14px;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .recommendation-item {
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .high-priority {
        background-color: rgba(239, 68, 68, 0.1);
        border-left-color: #EF4444;
    }
    .medium-priority {
        background-color: rgba(245, 158, 11, 0.1);
        border-left-color: #F59E0B;
    }
    .low-priority {
        background-color: rgba(59, 130, 246, 0.1);
        border-left-color: #3B82F6;
    }
    .recommendation-title {
        font-weight: bold;
        margin-bottom: 4px;
    }
    .recommendation-desc {
        color: #D1D5DB;
    }
    .severity-indicator {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-left: 8px;
        color: white;
    }
    .high-indicator {
        background-color: #EF4444;
    }
    .medium-indicator {
        background-color: #F59E0B; 
    }
    .low-indicator {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Page header with back navigation
st.markdown('<div class="page-header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 20])
with col1:
    st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
    if st.button('← ', key='back_button', help="Return to upload page"):
        st.switch_page("pages/1_upload_data.py")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="margin: 0; color: white;">Data Quality Assessment</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Check if data exists
if 'data' not in st.session_state or st.session_state.data is None:
    st.markdown(Config.empty_state(
        "No data loaded yet",
        "Upload a CSV or Excel file to begin — or load the bundled sample dataset from the Upload page."),
        unsafe_allow_html=True)
    if st.button('Go to Data Upload', use_container_width=True):
        st.switch_page("pages/1_upload_data.py")
    st.stop()

# Run the enhanced data quality assessment
quality_report = enhanced_data_quality(st.session_state.data)

# Display summary metrics in a card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(Config.metric_card("Rows", f'{quality_report["summary"]["row_count"]:,}'), unsafe_allow_html=True)

with col2:
    st.markdown(Config.metric_card("Columns", f'{quality_report["summary"]["column_count"]}'), unsafe_allow_html=True)

with col3:
    st.markdown(Config.metric_card("Duplicated Rows", f'{quality_report["summary"]["duplicated_rows"]:,}'),
                unsafe_allow_html=True)

with col4:
    st.markdown(Config.metric_card("Memory Usage", f'{quality_report["summary"]["memory_usage"]:.2f} MB'),
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Group recommendations by severity
if quality_report['recommendations']:
    # Escape user-derived strings (column names, and descriptions that embed cell
    # values) before they are rendered into raw HTML, to prevent XSS from a
    # malicious column name or value. Build copies -- do NOT mutate the cached report.
    recs = [
        {
            **r,
            'column': html.escape(str(r['column'])) if r.get('column') is not None else None,
            'description': html.escape(str(r.get('description', ''))),
        }
        for r in quality_report['recommendations']
    ]
    high_priority = [rec for rec in recs if rec['severity'] == 'high']
    medium_priority = [rec for rec in recs if rec['severity'] == 'medium']
    low_priority = [rec for rec in recs if rec['severity'] == 'low']

    # Recommendations card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

    # Column search feature
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    search_query = st.text_input("Search recommendations by column name:", placeholder="Enter column name...")
    st.markdown('</div>', unsafe_allow_html=True)

    # Get unique column names for autocomplete (excluding None values)
    column_names = set(rec['column'] for rec in recs if rec['column'] is not None)

    # Create filtered recommendations based on search
    if search_query:
        # Case-insensitive search
        search_query_lower = search_query.lower()

        # Filter recommendations containing the search term in column name
        search_results = [
            rec for rec in recs
            if rec['column'] and search_query_lower in rec['column'].lower()
        ]

        # Create a new tab set with search results and the regular tabs
        search_tab, tab1, tab2, tab3 = st.tabs([
            f"Search Results ({len(search_results)})",
            f"High Priority ({len(high_priority)})",
            f"Medium Priority ({len(medium_priority)})",
            f"Low Priority ({len(low_priority)})"
        ])

        with search_tab:
            if search_results:
                st.markdown(
                    f'<div class="search-results-count">Found {len(search_results)} recommendations for columns matching "{html.escape(search_query)}"</div>',
                    unsafe_allow_html=True)

                # Group search results by severity for better organization
                for i, rec in enumerate(search_results):
                    # Determine severity class for styling
                    severity_class = f"{rec['severity']}-priority"
                    severity_indicator = f"<span class='severity-indicator {rec['severity']}-indicator'>{rec['severity'].upper()}</span>"

                    column_info = f"<span class='recommendation-title'>Column: {rec['column']}</span> {severity_indicator}<br/>" if \
                    rec['column'] else ""

                    st.markdown(f"""
                    <div class="recommendation-item {severity_class}">
                        {column_info}
                        <div class="recommendation-desc">{rec['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="no-results">
                    No recommendations found for columns matching "{html.escape(search_query)}".
                </div>
                """, unsafe_allow_html=True)

                # Show column name suggestions if no results found
                if column_names:
                    st.markdown("### Available columns:")
                    st.write(", ".join(sorted(column_names)))
    else:
        # Regular tabs without search
        tab1, tab2, tab3 = st.tabs([
            f"High Priority ({len(high_priority)})",
            f"Medium Priority ({len(medium_priority)})",
            f"Low Priority ({len(low_priority)})"
        ])

    # High priority tab
    with tab1:
        if high_priority:
            for i, rec in enumerate(high_priority):
                column_info = f"<span class='recommendation-title'>Column: {rec['column']}</span><br/>" if rec[
                    'column'] else ""
                st.markdown(f"""
                <div class="recommendation-item high-priority">
                    {column_info}
                    <div class="recommendation-desc">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high priority issues found. Great job!")

    # Medium priority tab
    with tab2:
        if medium_priority:
            for i, rec in enumerate(medium_priority):
                column_info = f"<span class='recommendation-title'>Column: {rec['column']}</span><br/>" if rec[
                    'column'] else ""
                st.markdown(f"""
                <div class="recommendation-item medium-priority">
                    {column_info}
                    <div class="recommendation-desc">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No medium priority issues found.")

    # Low priority tab
    with tab3:
        if low_priority:
            for i, rec in enumerate(low_priority):
                column_info = f"<span class='recommendation-title'>Column: {rec['column']}</span><br/>" if rec[
                    'column'] else ""
                st.markdown(f"""
                <div class="recommendation-item low-priority">
                    {column_info}
                    <div class="recommendation-desc">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No low priority issues found.")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success("No data quality issues found. Your dataset looks great!")
    st.markdown('</div>', unsafe_allow_html=True)

# Navigation buttons at the bottom
st.markdown('<div style="display: flex; justify-content: flex-end; margin-top: 2rem;">', unsafe_allow_html=True)
if st.button('Continue to Preprocessing', key='continue_btn'):
    st.switch_page("pages/3_preprocessing.py")
st.markdown('</div>', unsafe_allow_html=True)