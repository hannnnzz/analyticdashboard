import streamlit as st
import pandas as pd
from components.style import apply_style, page_header, section_label, divider

def render(df_business, df_review=None, df_summary=None):
    apply_style()
    page_header("Business Summary", "Ringkasan ulasan per bisnis dataset Yelp berbasis NLP")

    # ── Merge dengan nama bisnis ──────────────────────────────────────────
    df = df_summary.merge(
        df_business[["business_id", "name", "city", "state", "stars", "review_count", "categories", "is_open"]],
        on="business_id", how="left"
    )

    # ── Search & Select Bisnis ────────────────────────────────────────────
    section_label("Pilih Bisnis")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        cat_opts = ["All"] + sorted(df["categories"].dropna().unique().tolist())
        cat_filter = st.selectbox("Filter Kategori", cat_opts)
    with col2:
        status_filter = st.selectbox("Status", ["All", "Open", "Closed"])
    with col3:
        city_opts = ["All"] + sorted(df["city"].dropna().unique().tolist())
        city_filter = st.selectbox("Filter Kota", city_opts)

    df_filtered = df.copy()
    if cat_filter != "All":
        df_filtered = df_filtered[df_filtered["categories"].str.contains(cat_filter, case=False, na=False)]
    if status_filter != "All":
        status_val = 1 if status_filter == "Open" else 0
        df_filtered = df_filtered[df_filtered["is_open"].apply(lambda x: int(float(x)) if pd.notna(x) else -1) == status_val]
    if city_filter != "All":
        df_filtered = df_filtered[df_filtered["city"] == city_filter]

    biz_options = df_filtered["name"].dropna().unique().tolist()

    if not biz_options:
        st.warning("Tidak ada bisnis yang cocok.")
        return

    selected_name = st.selectbox("Pilih Bisnis", sorted(biz_options))
    selected_row = df_filtered[df_filtered["name"] == selected_name].iloc[0]

    divider()

    # ── KPI Bisnis ────────────────────────────────────────────────────────
    section_label("Business Profile")

    kpi_style = "padding:16px;border-radius:10px;background:#1e2130;border:1px solid #2e3250;text-align:center;min-height:100px;display:flex;flex-direction:column;justify-content:center;"
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">City</div>
            <div style="font-size:22px;font-weight:700;color:#60a5fa;">{selected_row.get('city', '-')}</div>
            <div style="font-size:12px;color:#6b7280;">{selected_row.get('state', '-')}</div>
            <div style="font-size:4px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        stars_val = selected_row.get('stars', '-')
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Avg Stars</div>
            <div style="font-size:26px;font-weight:700;color:#facc15;">{'⭐ ' + str(stars_val) if stars_val != '-' else '-'}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        rc = selected_row.get('review_count', 0)
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Total Reviews</div>
            <div style="font-size:26px;font-weight:700;color:#34d399;">{f'{int(rc):,}' if rc else '-'}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        is_open_raw = selected_row.get('is_open', None)
        try:
            is_open = int(float(is_open_raw))
        except (ValueError, TypeError):
            is_open = None
        open_label = "Open" if is_open == 1 else "Closed" if is_open == 0 else "-"
        open_color = "#34d399" if is_open == 1 else "#f87171" if is_open == 0 else "#9ca3af"
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Status</div>
            <div style="font-size:26px;font-weight:700;color:{open_color};">{open_label}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)
    # Kategori
    if pd.notna(selected_row.get("categories")):
        st.markdown(f"""<div style="margin-top:10px;padding:10px 14px;border-radius:8px;background:#1e2130;border:1px solid #2e3250;">
            <span style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;">Categories &nbsp;</span>
            <span style="color:#a5b4fc;font-size:13px;">{selected_row['categories']}</span>
        </div>""", unsafe_allow_html=True)

    divider()

    # ── Summary Card ──────────────────────────────────────────────────────
    section_label("Model Generated Summary")

    summary_text = selected_row.get("summary", "Summary tidak tersedia.")
    st.markdown(f"""
    <div style="padding:20px 24px;border-radius:12px;background:#1e2130;border:1px solid #2e3250;
                line-height:1.8;color:#e5e7eb;font-size:15px;">
        <span style="font-size:28px;color:#818cf8;line-height:1;">"</span>
        {summary_text}
        <span style="font-size:28px;color:#818cf8;line-height:1;">"</span>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Review Table untuk bisnis ini ─────────────────────────────────────
    if df_review is not None and "business_id" in df_review.columns:
        section_label("Individual Reviews")

        biz_id = selected_row["business_id"]
        df_rev = df_review[df_review["business_id"] == biz_id].copy()

        if not df_rev.empty:
            # Mini KPI review
            r1, r2, r3 = st.columns(3)
            with r1:
                avg_sent = df_rev["sentiment_score"].mean() if "sentiment_score" in df_rev.columns else None
                st.metric("Avg Sentiment Score", f"{avg_sent:.3f}" if avg_sent else "-")
            with r2:
                top_emo = df_rev["top_emotion"].value_counts().idxmax() if "top_emotion" in df_rev.columns else "-"
                st.metric("Top Emotion", top_emo.title() if top_emo != "-" else "-")
            with r3:
                pct_pos = (df_rev["sentiment_label"] == "POSITIVE").sum() / len(df_rev) * 100 if "sentiment_label" in df_rev.columns else None
                st.metric("% Positive", f"{pct_pos:.1f}%" if pct_pos else "-")

            show_cols = [c for c in ["stars", "date", "sentiment_label", "sentiment_score", "top_emotion", "text"] if c in df_rev.columns]
            rename_map = {
                "stars": "Stars", "date": "Date", "sentiment_label": "Sentiment",
                "sentiment_score": "Score", "top_emotion": "Emotion", "text": "Review"
            }
            st.dataframe(
                df_rev[show_cols].rename(columns=rename_map).head(100).reset_index(drop=True),
                use_container_width=True, height=350
            )
        else:
            st.info("Tidak ada review untuk bisnis ini.")