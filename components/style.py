import streamlit as st


def apply_style():
    st.markdown("""
    <style>
    /* ── Font & background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; color: #e5e7eb; }

    /* ── KPI row (new) ── */
    .kpi-row {
        display: flex;
        gap: 16px;
        justify-content: space-between;
        align-items: stretch;
        flex-wrap: wrap;
        margin-bottom: 12px;
        
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        /* responsive width: 5 cards in a row on wide screens, wrap on small screens */
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        flex: 1 1 calc(20% - 16px);
        min-width: 160px;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.12);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #e5e7eb;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .kpi-accent  { color: #818cf8; }
    .kpi-green   { color: #34d399; }
    .kpi-yellow  { color: #fbbf24; }
    .kpi-red     { color: #f87171; }
    .kpi-blue    { color: #60a5fa; }

    /* top-category (big card) */
    .kpi-topcat {
        flex: 1 1 100%;
        display: inline-block;
        text-align: left;
        padding: 28px 24px;
    }
    .kpi-topcat .kpi-value {
        font-size: 22px;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 28px 0 4px 0;
        padding-left: 2px;
    }

    /* ── Divider ── */
    .dash-divider {
        border: none;
        border-top: 1px solid #2e3250;
        margin: 20px 0;
    }

    /* ── Chart card ── */
    .chart-card {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* grid helper for charts: 2 columns on wide screens, 1 column on small */
    .chart-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
    }
    @media (max-width: 900px) {
        .chart-grid { grid-template-columns: 1fr; }
        .kpi-card { flex: 1 1 calc(50% - 16px); }
    }
    @media (max-width: 640px) {
        .kpi-card { flex: 1 1 100%; }
    }

    /* ── Page title ── */
    .page-title {
        font-size: 22px;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 24px;
    }

    /* ── Streamlit overrides ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label { color: #9ca3af !important; font-size: 12px !important; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #e5e7eb !important; font-size: 26px !important; font-weight: 700 !important;
    }

    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)


# Plotly dark theme config — pakai di semua chart
PLOTLY_THEME = "plotly_dark"
CHART_BG    = "#1e2130"
PAPER_BG    = "#1e2130"
GRID_COLOR  = "#2e3250"
FONT_COLOR  = "#9ca3af"
ACCENT      = "#818cf8"

COLOR_SEQ = [
    "#818cf8", "#34d399", "#fbbf24", "#f87171",
    "#60a5fa", "#a78bfa", "#fb923c", "#2dd4bf",
]


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, family="Inter"),
        margin=dict(l=16, r=16, t=40, b=16),
        title_font=dict(size=14, color="#e5e7eb"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR)),
        colorway=COLOR_SEQ,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig