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

    # Theme colors (kept in sync with the CSS design tokens below)
    COLORS = {
        "background": "#0b1020",
        "card_bg": "rgba(255, 255, 255, 0.045)",
        "primary": "#6366F1",
        "primary_hover": "#818CF8",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "text": "#F8FAFC",
        "text_secondary": "#94A3B8",
    }

    # Global design system + theme (injected on every page).
    @classmethod
    def get_css(cls):
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* ============ DESIGN TOKENS ============ */
        :root {
            /* color */
            --bg: #0b1020;
            --surface: rgba(255, 255, 255, 0.045);
            --surface-2: rgba(255, 255, 255, 0.07);
            --border: rgba(255, 255, 255, 0.09);
            --border-strong: rgba(255, 255, 255, 0.16);
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --text-faint: #64748b;
            --primary: #6366f1;
            --primary-2: #8b5cf6;
            --accent-grad: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #3b82f6 100%);
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            /* type */
            --font-body: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
            --font-display: 'Space Grotesk', 'Inter', sans-serif;
            /* space */
            --space-2: 8px; --space-3: 12px; --space-4: 16px;
            --space-5: 24px; --space-6: 32px; --space-7: 48px;
            /* radius */
            --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px; --radius-xl: 22px;
            /* shadow + glow */
            --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.35);
            --shadow-lg: 0 18px 48px rgba(0, 0, 0, 0.45);
            --glow: 0 0 24px rgba(99, 102, 241, 0.45);
            /* motion */
            --ease: cubic-bezier(0.4, 0, 0.2, 1);
            --dur-fast: 140ms; --dur: 240ms; --dur-slow: 420ms;
        }

        /* ============ STREAMLIT CHROME ============ */
        header[data-testid="stHeader"] { display: none !important; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; }

        /* ============ APP BACKGROUND (deep navy + aurora glow) ============ */
        .stApp {
            background:
                radial-gradient(1200px 620px at 14% -12%, rgba(99, 102, 241, 0.18), transparent 60%),
                radial-gradient(1000px 520px at 92% -4%, rgba(139, 92, 246, 0.14), transparent 55%),
                var(--bg);
            color: var(--text);
            font-family: var(--font-body);
        }

        /* ============ TYPOGRAPHY ============ */
        .stApp, .stApp p, .stApp li, .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp button, .stApp input, .stApp textarea, .stApp select {
            font-family: var(--font-body);
        }
        h1, h2, h3, h4, .title, .section-title {
            font-family: var(--font-display);
            letter-spacing: -0.02em;
        }

        /* hero title — gradient-clipped text */
        .title {
            font-size: clamp(2.3rem, 5vw, 3.4rem);
            font-weight: 700;
            line-height: 1.05;
            background: var(--accent-grad);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 var(--space-3) 0;
        }
        .subtitle { color: var(--text-dim); font-size: 1.08rem; line-height: 1.6; }
        .center-content { text-align: center; margin-bottom: var(--space-6); padding-top: var(--space-4); }

        /* ============ GLASS CARDS ============ */
        .card, .content-area {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: var(--space-5);
            margin-bottom: var(--space-5);
            backdrop-filter: blur(14px) saturate(140%);
            -webkit-backdrop-filter: blur(14px) saturate(140%);
            box-shadow: var(--shadow-md);
        }
        /* hide orphan empty wrappers — Streamlit renders a markdown <div class="card">
           as a standalone empty node instead of wrapping the widgets that follow it. */
        .card:empty, .content-area:empty, .metric-container:empty { display: none !important; }
        .section-title { font-size: 1.2rem; font-weight: 600; color: var(--text); margin-bottom: var(--space-4); }
        .page-header {
            padding: var(--space-3) 0 var(--space-4);
            border-bottom: 1px solid var(--border);
            margin-bottom: var(--space-6);
            display: flex; justify-content: space-between; align-items: center;
        }

        /* ============ BUTTONS (gradient + hover lift + glow) ============ */
        /* target the button's own data-testid — robust against tooltip/wrapper divs
           that break a `div.stButton > button` direct-child selector. */
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]) {
            background: var(--accent-grad) !important;
            color: #fff !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 0.55rem 1.3rem !important;
            font-weight: 600 !important;
            font-family: var(--font-body) !important;
            box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
            transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease), filter var(--dur) var(--ease);
        }
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]):hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(99, 102, 241, 0.5), var(--glow);
            filter: brightness(1.07);
        }
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]):active { transform: translateY(0); }
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]):disabled,
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]):disabled:hover {
            opacity: 0.45 !important; filter: grayscale(0.4); box-shadow: none; transform: none;
        }

        /* ============ METRIC CARDS ============ */
        .metric-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: var(--space-4);
            margin-bottom: var(--space-3);
        }
        .metric-label {
            color: var(--text-dim); font-size: 0.8rem; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.05em;
        }
        .metric-value { color: var(--text); font-size: 1.55rem; font-weight: 700; font-family: var(--font-display); }

        /* ============ CHECKLIST ============ */
        .checkbox-item { display: flex; align-items: center; gap: var(--space-2); margin-bottom: var(--space-3); color: var(--text); }
        .green-check { color: var(--success); }

        /* ============ U1: WIDGETS ============ */

        /* text / number / textarea inputs */
        .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-faint) !important; }

        /* selectbox / multiselect */
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
        }
        .stSelectbox div[data-baseweb="select"] > div:focus-within,
        .stMultiSelect div[data-baseweb="select"] > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        }
        ul[data-baseweb="menu"], div[data-baseweb="popover"] ul {
            background: #131a2e !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: var(--radius-md) !important;
        }

        /* tabs: gradient active underline */
        .stTabs [data-baseweb="tab-list"] { gap: var(--space-3); border-bottom: 1px solid var(--border); }
        .stTabs button[data-baseweb="tab"] { color: var(--text-dim); font-weight: 600; transition: color var(--dur) var(--ease); }
        .stTabs button[data-baseweb="tab"]:hover { color: var(--text); }
        .stTabs button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
        .stTabs [data-baseweb="tab-highlight"] { background: var(--accent-grad) !important; height: 3px !important; border-radius: 3px; }
        .stTabs [data-baseweb="tab-border"] { background: var(--border) !important; }

        /* slider thumb glow */
        .stSlider [role="slider"] { box-shadow: var(--glow) !important; }

        /* expander glass */
        details[data-testid="stExpander"], .stExpander details {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden;
        }
        .stExpander summary:hover { background: var(--surface-2) !important; }

        /* alerts as coloured glass */
        .stAlert, div[data-testid="stAlert"] {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
            backdrop-filter: blur(10px);
        }
        .stSuccess { border-left: 3px solid var(--success) !important; }
        .stInfo { border-left: 3px solid var(--primary) !important; }
        .stWarning { border-left: 3px solid var(--warning) !important; }
        .stError { border-left: 3px solid var(--error) !important; }

        /* file-drop zone: glow on hover */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface) !important;
            border: 1.5px dashed var(--border-strong) !important;
            border-radius: var(--radius-lg) !important;
            transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease), background var(--dur) var(--ease);
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--primary) !important;
            background: var(--surface-2) !important;
            box-shadow: var(--glow);
        }

        /* dataframe / table */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden;
        }

        /* caption + inline code */
        [data-testid="stCaptionContainer"] { color: var(--text-faint) !important; }
        code { background: var(--surface) !important; border: 1px solid var(--border); border-radius: 6px; padding: 0.1em 0.4em; }
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
