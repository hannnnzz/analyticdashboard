import streamlit as st
from components.style import apply_style, divider

def render():
    apply_style()

    st.markdown("""
    <div style="padding: 48px 0 24px 0; text-align: center;">
        <div style="font-size: 11px; letter-spacing: 0.2em; color: #818cf8; 
                    text-transform: uppercase; margin-bottom: 16px; font-weight: 600;">
            ✦ Yelp Business Intelligence Platform
        </div>
        <h1 style="font-size: 48px; font-weight: 800; margin: 0;
                   background: linear-gradient(135deg, #e5e7eb 0%, #818cf8 50%, #60a5fa 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Analytics Dashboard
        </h1>
        <p style="color: #6b7280; font-size: 16px; margin-top: 16px; max-width: 560px;
                  margin-left: auto; margin-right: auto; line-height: 1.7;">
            Eksplorasi data bisnis Yelp melalui pendekatan Business Intelligence, 
            Natural Language Processing, dan Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Module Cards ───────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:11px;letter-spacing:0.15em;color:#6b7280;text-transform:uppercase;font-weight:600;margin-bottom:20px;'>Modul Tersedia</div>", unsafe_allow_html=True)

    modules = [
        {
            "title": "Business Intelligence",
            "desc": "Market overview, analisis reputasi & popularitas, serta strategi diferensiasi bisnis.",
            "tags": ["Market Overview", "Reputation", "Strategy"],
            "color": "#60a5fa",
            "bg": "#0d1b2e",
        },
        {
            "title": "Natural Language Processing",
            "desc": "Analisis sentimen & emosi dari review, serta ringkasan otomatis per bisnis.",
            "tags": ["Sentiment", "Emotion", "Summary"],
            "color": "#a78bfa",
            "bg": "#1a1035",
        },
        {
            "title": "User Segmentation",
            "desc": "Clustering pengguna berbasis perilaku, visualisasi social graph, dan prediksi segmen.",
            "tags": ["Clustering", "Social Graph", "Prediction"],
            "color": "#34d399",
            "bg": "#0d2b1f",
        },
        {
            "title": "Churn Analysis",
            "desc": "Identifikasi user berisiko churn berdasarkan aktivitas review dan pola check-in.",
            "tags": ["User Churn", "Checkin Behavior"],
            "color": "#fb923c",
            "bg": "#2b1a0d",
        },
        {
            "title": "Recommendation System",
            "desc": "Rekomendasi bisnis personal menggunakan Hybrid Neural Collaborative Filtering.",
            "tags": ["Hybrid NCF", "Personalized"],
            "color": "#fbbf24",
            "bg": "#2b2010",
        },
    ]

    # Baris 1: 3 kolom
    cols1 = st.columns(3)
    for col, mod in zip(cols1, modules[:3]):
        with col:
            tags_html = "".join([
                f'<span style="font-size:9px;color:{mod["color"]};background:{mod["color"]}22;'
                f'padding:2px 8px;border-radius:999px;border:1px solid {mod["color"]}44;'
                f'margin-right:4px;font-weight:600;">{t}</span>'
                for t in mod["tags"]
            ])
            st.markdown(f"""
            <div style="padding:24px;border-radius:16px;background:linear-gradient(135deg,{mod["bg"]},{mod["bg"]}cc);
                        border:1px solid {mod["color"]}33;margin-bottom:16px;min-height:200px;
                        box-shadow:0 4px 24px {mod["color"]}11;">
                <div style="font-size:20px;font-weight:700;color:#e5e7eb;margin-bottom:8px;">{mod["title"]}</div>
                <div style="font-size:12px;color:#9ca3af;line-height:1.6;margin-bottom:14px;">{mod["desc"]}</div>
                <div>{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

    # Baris 2: 2 kolom (center)
    _, c1, c2, _ = st.columns([0.5, 1, 1, 0.5])
    for col, mod in zip([c1, c2], modules[3:]):
        with col:
            tags_html = "".join([
                f'<span style="font-size:9px;color:{mod["color"]};background:{mod["color"]}22;'
                f'padding:2px 8px;border-radius:999px;border:1px solid {mod["color"]}44;'
                f'margin-right:4px;font-weight:600;">{t}</span>'
                for t in mod["tags"]
            ])
            st.markdown(f"""
            <div style="padding:24px;border-radius:16px;background:linear-gradient(135deg,{mod["bg"]},{mod["bg"]}cc);
                        border:1px solid {mod["color"]}33;margin-bottom:16px;min-height:200px;
                        box-shadow:0 4px 24px {mod["color"]}11;">
                <div style="font-size:20px;font-weight:700;color:#e5e7eb;margin-bottom:8px;">{mod["title"]}</div>
                <div style="font-size:12px;color:#9ca3af;line-height:1.6;margin-bottom:14px;">{mod["desc"]}</div>
                <div>{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
        <span style="font-size:11px;color:#374151;">Built with</span>
        <span style="font-size:11px;color:#6b7280;"> Streamlit · PyTorch · Plotly · Scikit-learn</span>
    </div>
    """, unsafe_allow_html=True)