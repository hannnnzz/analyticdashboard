import streamlit as st
import pandas as pd
from components.style import apply_style, page_header, section_label, divider
from data.loader_wordcloud import load_wordcloud
from components.charts_nlp import (
    chart_sentiment_distribution, chart_sentiment_score_hist,
    chart_sentiment_over_time, chart_stars_vs_sentiment,
    chart_word_frequency,
)


def render(df_review):
    apply_style()
    page_header("Sentiment Analysis", "Analisis sentimen dari pengguna Yelp menggunakan DeepLearning")

    # ── Pakai langsung dari df_review ─────────────────────────────────────────
    df = df_review.copy()

    required = ["sentiment_label", "sentiment_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Kolom tidak tersedia: {missing}")
        return

    if df.empty:
        st.warning("Data tidak tersedia.")
        return

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    section_label("Overview")
    total     = len(df)
    pos       = (df["sentiment_label"] == "POSITIVE").sum()
    neg       = (df["sentiment_label"] == "NEGATIVE").sum()
    avg_score = df["sentiment_score"].mean()
    pct_pos   = pos / total * 100
    pct_neg   = neg / total * 100

    k1, k2, k3, k4 = st.columns(4)
    kpi_style = "padding:16px;border-radius:10px;background:#1e2130;border:1px solid #2e3250;text-align:center;min-height:100px;display:flex;flex-direction:column;justify-content:center;"

    with k1:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Total Reviews</div>
            <div style="font-size:26px;font-weight:700;color:#60a5fa;">{total:,}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div> 
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Positive</div>
            <div style="font-size:26px;font-weight:700;color:#34d399;">{pos:,}</div>
            <div style="font-size:12px;color:#6b7280;">{pct_pos:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Negative</div>
            <div style="font-size:26px;font-weight:700;color:#f87171;">{neg:,}</div>
            <div style="font-size:12px;color:#6b7280;">{pct_neg:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Avg Sentiment Score</div>
            <div style="font-size:26px;font-weight:700;color:#818cf8;">{avg_score:.3f}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div> 
        </div>""", unsafe_allow_html=True)

    divider()

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Trend", "By Rating", "Word Frequency"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Sentiment Label Distribution")
            fig = chart_sentiment_distribution(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")
        with c2:
            section_label("Avg Sentiment Score per Star Rating")
            fig = chart_sentiment_score_hist(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with tab2:
        section_label("Sentiment Trend Over Time")
        fig = chart_sentiment_over_time(df)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("Kolom date tidak tersedia")

    with tab3:
        section_label("Sentiment per Star Rating")
        fig = chart_stars_vs_sentiment(df)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("Kolom stars tidak tersedia")

    with tab4:
        section_label("Word Cloud")
        n_words = st.slider("Top N words", min_value=20, max_value=200, value=100, step=10)
        fig = chart_word_frequency(n=n_words)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data tidak tersedia")

    divider()

    # ── Tabel Review ──────────────────────────────────────────────────────────
    section_label("Review Table")
    col_filter, col_star = st.columns([2, 2])
    with col_filter:
        sentiment_opt = st.selectbox("Filter Sentiment", ["All", "POSITIVE", "NEGATIVE"])
    with col_star:
        if "stars" in df.columns:
            star_opt = st.multiselect("Filter Stars", sorted(df["stars"].dropna().unique()),
                                      default=sorted(df["stars"].dropna().unique()))
        else:
            star_opt = None

    df_table = df.copy()
    if sentiment_opt != "All":
        df_table = df_table[df_table["sentiment_label"] == sentiment_opt]
    if star_opt and "stars" in df_table.columns:
        df_table = df_table[df_table["stars"].isin(star_opt)]

    show_cols = [c for c in ["review_id", "stars", "date", "sentiment_label",
                          "sentiment_score", "text"] if c in df_table.columns]

    rename_cols = {
        "review_id": "Review ID",
        "stars": "Stars",
        "date": "Date",
        "sentiment_label": "Sentiment Label",
        "sentiment_score": "Sentiment Score",
        "text": "Text",
    }

    st.dataframe(df_table[show_cols].rename(columns=rename_cols).head(200).reset_index(drop=True),
                use_container_width=True, height=350)