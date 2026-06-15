# UI Overhaul Workplan — a premium, animated Streamlit UI

Goal: take the app from "default Streamlit + a dark theme" to a polished, modern,
animated product UI — with a high-fidelity visual system and tasteful advanced
animations — within what Streamlit actually allows.

## What's achievable in Streamlit (honest constraints)
- ✅ **Global CSS** injected via `st.markdown(unsafe_allow_html=True)` — full control
  of colours, type, spacing, depth, and **CSS animations/transitions**.
- ✅ **`.streamlit/config.toml [theme]`** — aligns Streamlit's own widgets to the palette/font.
- ✅ **`st.components.v1.html`** (sandboxed iframe) — real **JS/WebGL** runs here, so a
  particle / shader "engine" hero is possible with **no new Python dependency**.
- ⚠️ Inline `<script>` in `st.markdown` is sanitized (won't run) → JS effects live in the iframe.
- ⚠️ Deep widget styling targets Streamlit `data-testid`s (can shift across versions) → use stable hooks, keep it resilient.

## Phases (each verified in a real browser; permission asked after each)

| # | Phase | What it delivers | Effort |
|---|-------|------------------|--------|
| **U0** | **Design system & theme** | A token-based CSS foundation (colour/type/space/radius/shadow/motion CSS variables) + a premium Google font + `[theme]` in config.toml. The base everything builds on. | M |
| **U1** | **Global component styling** | Cohesive styling of every widget: gradient buttons w/ hover-lift + glow, focus rings on inputs/selects/sliders, animated tab underline, glass metric cards, themed dataframes/expanders/alerts, a glowing file-drop zone, glassmorphism cards + layered depth. | L |
| **U2** | **Motion & micro-interactions** | Staggered fade/slide entrance on load, hover micro-interactions, button shine/ripple, loading shimmer/skeletons, an animated aurora/gradient background — all CSS, with a `prefers-reduced-motion` accessibility guard. | M–L |
| **U3** | **"Realistic engine" hero (landing)** | Redesign the landing into a showcase: a `components.v1.html` iframe running an interactive **WebGL/particle** background (CDN three.js/particles, self-contained), a glass hero card, animated headline + gradient CTA. The wow moment. | L |
| **U4** | **Workflow pages + progress** | Apply the system across all 6 pages; add a **workflow step indicator** (1 Upload → … → 6 ML) so users feel progress; consistent headers, empty states, loading states. | M–L |
| **U5** | **Advanced polish & QA** | Favicon/branding, page icons, animated success/empty states, responsive tweaks, contrast/a11y pass, reduced-motion, cross-page browser QA, performance (no jank). | M |

## Principles
- One cohesive design language (tokens), not per-page one-offs.
- Motion with **purpose** (feedback, hierarchy, delight) — never gratuitous jank.
- Dependency-light: CSS + built-in `components.v1.html`. (Optional `streamlit-lottie`
  only if you want Lottie — flagged, not assumed.)
- Accessible: contrast, focus-visible, `prefers-reduced-motion`.
- Resilient selectors so a Streamlit update won't shatter the theme.
