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

        /* ============ U2: MOTION & MICRO-INTERACTIONS ============ */

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
        @keyframes btn-shine { from { left: -120%; } to { left: 150%; } }
        @keyframes shimmer { from { background-position: -468px 0; } to { background-position: 468px 0; } }

        /* aurora is the static .stApp background (from the design tokens above) —
           a continuously animated full-viewport blur caused scroll jank, so the
           background stays still and motion lives in interactions instead. */

        /* page entrance — top-level blocks cascade up (precise path targets only
           top-level blocks, so React keeps them stable → no replay on rerun) */
        [data-testid="stMainBlockContainer"] { animation: fadeIn 0.45s var(--ease) both; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div {
            animation: fadeInUp 0.5s var(--ease) both;
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:nth-child(1) { animation-delay: 0.03s; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.07s; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.11s; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.15s; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:nth-child(5) { animation-delay: 0.19s; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:nth-child(6) { animation-delay: 0.23s; }

        /* button shine sweep on hover */
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]) { position: relative; overflow: hidden; }
        button[data-testid^="stBaseButton"]:not([data-testid*="header"])::after {
            content: ""; position: absolute; top: 0; left: -120%; width: 55%; height: 100%;
            background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.38), transparent);
            transform: skewX(-18deg); pointer-events: none;
        }
        button[data-testid^="stBaseButton"]:not([data-testid*="header"]):hover::after { animation: btn-shine 0.85s var(--ease); }

        /* card / metric hover micro-interactions */
        .card, .metric-container { transition: transform var(--dur) var(--ease), border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease); }
        .card:not(:empty):hover { transform: translateY(-3px); border-color: var(--border-strong); box-shadow: var(--shadow-lg), 0 0 22px rgba(99, 102, 241, 0.18); }
        .metric-container:not(:empty):hover { transform: translateY(-2px); border-color: var(--border-strong); }

        /* loading shimmer skeleton utility + spinner glow */
        .skeleton {
            background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.10) 37%, rgba(255,255,255,0.04) 63%);
            background-size: 936px 100%;
            animation: shimmer 1.4s linear infinite;
            border-radius: var(--radius-sm);
            min-height: 1rem;
        }
        [data-testid="stSpinner"] svg { filter: drop-shadow(0 0 6px rgba(99, 102, 241, 0.6)); }

        /* ============ U4: WORKFLOW STEP INDICATOR ============ */
        .stepper { display: flex; align-items: center; margin: 0 0 var(--space-6); }
        .step { display: flex; align-items: center; gap: 0.5rem; flex: 0 0 auto; }
        .step-dot {
            width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
            font-size: 0.85rem; font-weight: 700; font-family: var(--font-display);
            background: var(--surface-2); color: var(--text-dim); border: 1px solid var(--border);
            transition: all var(--dur) var(--ease);
        }
        .step-label { font-size: 0.82rem; color: var(--text-dim); font-weight: 500; white-space: nowrap; }
        .step-line { flex: 1 1 auto; height: 2px; min-width: 12px; background: var(--border); margin: 0 0.6rem; border-radius: 2px; }
        .step-line.done { background: linear-gradient(90deg, var(--success), var(--primary)); }
        .step.done .step-dot { color: var(--success); border-color: rgba(16, 185, 129, 0.45); }
        .step.current .step-dot { background: var(--accent-grad); color: #fff; border: none; box-shadow: var(--glow); }
        .step.current .step-label { color: var(--text); font-weight: 700; }
        @media (max-width: 720px) { .step-label { display: none; } }

        /* U4: empty-state panel */
        .empty-state {
            text-align: center; padding: var(--space-7) var(--space-5);
            border: 1px dashed var(--border-strong); border-radius: var(--radius-lg);
            background: var(--surface); color: var(--text-dim);
        }
        .empty-state .es-icon { font-size: 2.4rem; margin-bottom: var(--space-3); display: block; }
        .empty-state .es-title { color: var(--text); font-weight: 600; font-size: 1.1rem; margin-bottom: var(--space-2); font-family: var(--font-display); }

        /* ============ U5: POLISH & A11Y ============ */
        /* keyboard focus rings (only on keyboard nav, not mouse clicks) */
        button[data-testid^="stBaseButton"]:focus-visible,
        [data-testid="stSidebarNavLink"]:focus-visible,
        .stTabs button[data-baseweb="tab"]:focus-visible,
        a:focus-visible {
            outline: 2px solid var(--primary) !important;
            outline-offset: 3px;
            border-radius: var(--radius-sm);
        }
        html { scroll-behavior: smooth; }
        ::selection { background: rgba(99, 102, 241, 0.38); color: #fff; }
        /* themed scrollbar */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 6px; border: 2px solid transparent; background-clip: padding-box; }
        ::-webkit-scrollbar-thumb:hover { background: var(--border-strong); }
        * { scrollbar-color: var(--surface-2) transparent; }
        /* success state eases in */
        .stSuccess { animation: fadeInUp 0.4s var(--ease) both; }

        /* accessibility: honor reduced-motion — neutralize all motion */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        """

    # Workflow steps for the progress indicator (one per page, 1..6).
    WORKFLOW_STEPS = ["Upload", "Quality", "Prepare", "Visualize", "Dashboard", "Model"]

    @classmethod
    def stepper(cls, current):
        """Render the workflow step indicator. `current` is 1-based (1..6)."""
        out = ['<div class="stepper">']
        for i, name in enumerate(cls.WORKFLOW_STEPS, start=1):
            if i > 1:
                out.append('<div class="step-line {}"></div>'.format("done" if i <= current else ""))
            state = "done" if i < current else ("current" if i == current else "")
            mark = "✓" if i < current else str(i)
            out.append('<div class="step {}"><div class="step-dot">{}</div>'
                       '<div class="step-label">{}</div></div>'.format(state, mark, name))
        out.append('</div>')
        return "".join(out)

    @staticmethod
    def metric_card(label, value):
        """A single-block glass metric card (label + value wrapped together, so the
        container actually wraps its content instead of rendering as an empty box)."""
        return ('<div class="metric-container"><span class="metric-label">{}</span>'
                '<div class="metric-value">{}</div></div>').format(label, value)

    @staticmethod
    def empty_state(title, message, icon="📂"):
        """A styled empty-state panel for pages that have no data yet."""
        return ('<div class="empty-state"><span class="es-icon">{}</span>'
                '<div class="es-title">{}</div><div>{}</div></div>').format(icon, title, message)


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
