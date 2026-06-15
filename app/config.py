# config.py
import os


class Config:
    # Application metadata
    APP_NAME = "Data Analysis Platform"
    VERSION = "1.0.0"

    # File settings
    ALLOWED_EXTENSIONS = ["csv", "xlsx", "xls"]
    MAX_FILE_SIZE_MB = 50  # keep within Streamlit Community Cloud's ~1 GB memory
    DEFAULT_PREVIEW_ROWS = 10

    # Theme colors
    COLORS = {
        "background": "#111827",  # Dark blue-gray background
        "card_bg": "rgba(255, 255, 255, 0.05)",  # Card background
        "primary": "#3B82F6",  # Primary blue for buttons
        "primary_hover": "#2563EB",  # Hover state for buttons
        "success": "#10B981",  # Green for success messages
        "warning": "#F59E0B",  # Yellow for warnings
        "error": "#EF4444",  # Red for errors
        "text": "#FFFFFF",  # Main text color
        "text_secondary": "#9CA3AF",  # Secondary text color
    }

    # CSS Styles (common styles applied to all pages)
    @classmethod
    def get_css(cls):
        return f"""
        <style>
            /* Eliminate top spacing and header elements */
            header {{display: none !important;}}
            .block-container {{
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                margin-top: 0 !important;
            }}

            /* App background */
            .stApp {{
                background-color: {cls.COLORS["background"]};
                color: {cls.COLORS["text"]};
            }}

            /* Hide Streamlit default elements */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}

            /* Page header */
            .page-header {{
                padding: 1rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            /* Card styling */
            .card {{
                background-color: {cls.COLORS["card_bg"]};
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }}

            /* Section titles */
            .section-title {{
                color: {cls.COLORS["text"]};
                font-size: 1.25rem;
                margin-bottom: 1rem;
            }}

            /* Button styling */
            div.stButton > button {{
                background-color: {cls.COLORS["primary"]};
                color: {cls.COLORS["text"]};
                border: none;
                border-radius: 6px;
                transition: all 0.3s ease;
            }}

            div.stButton > button:hover {{
                background-color: {cls.COLORS["primary_hover"]};
                box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
            }}

            /* Back button styling */
            .back-button {{
                background-color: transparent !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                color: white !important;
            }}

            .back-button:hover {{
                background-color: rgba(255, 255, 255, 0.1) !important;
                box-shadow: none !important;
            }}

            /* Success/info message styling */
            .stSuccess, .stInfo {{
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: none;
                padding: 0.75rem;
                border-radius: 6px;
            }}

            .stSuccess > div {{
                border-top-color: {cls.COLORS["success"]} !important;
            }}

            .stInfo > div {{
                border-top-color: {cls.COLORS["primary"]} !important;
            }}

            /* File uploader styling */
            .uploadedFile {{
                background-color: rgba(255, 255, 255, 0.1) !important;
                border-radius: 6px !important;
                padding: 0.5rem !important;
            }}

            /* Content area styling */
            .content-area {{
                background-color: {cls.COLORS["card_bg"]};
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }}

            /* Title styling */
            .title {{
                color: {cls.COLORS["text"]};
                font-size: 2.5rem;
                font-weight: bold;
            }}

            /* Subtitle styling */
            .subtitle {{
                color: {cls.COLORS["text_secondary"]};
                font-size: 1.1rem;
            }}

            /* Center content styling */
            .center-content {{
                text-align: center;
                margin-bottom: 1.5rem;
                padding-top: 1rem;
            }}

            /* Checkbox styling */
            .checkbox-item {{
                display: flex;
                align-items: center;
                margin-bottom: 0.5rem;
            }}

            .green-check {{
                color: {cls.COLORS["success"]};
                margin-right: 0.5rem;
            }}

            /* Data metrics */
            .metric-container {{
                background-color: {cls.COLORS["card_bg"]};
                border-radius: 6px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }}

            .metric-label {{
                color: {cls.COLORS["text_secondary"]};
                font-size: 0.875rem;
            }}

            .metric-value {{
                color: {cls.COLORS["text"]};
                font-size: 1.25rem;
                font-weight: bold;
            }}
        </style>
        """


# Path configuration. Anchored to this file's directory so they resolve no matter
# the working directory (e.g. Streamlit Cloud runs from the repo root with the
# main file at app/home.py).
class Paths:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
    STATIC_DIR = os.path.join(_APP_DIR, "static")
    TEMP_DIR = os.path.join(_APP_DIR, "temp")
    UPLOAD_DIR = os.path.join(_APP_DIR, "uploads")


# Feature flags - useful for enabling/disabling features
class FeatureFlags:
    ENABLE_DATA_QUALITY_SCORE = True
    ENABLE_DASHBOARD_EXPORT = True
    ENABLE_RECENT_FILES = False  # Feature coming soon