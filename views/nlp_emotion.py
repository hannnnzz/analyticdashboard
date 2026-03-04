import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.style import apply_style, page_header, section_label, divider, style_fig, COLOR_SEQ, FONT_COLOR, GRID_COLOR, CHART_BG

EMOTION_COLORS = {
    "admiration": "#818cf8",
    "amusement": "#fbbf24",
    "anger": "#f87171",
    "annoyance": "#fb923c",
    "approval": "#34d399",
    "caring": "#2dd4bf",
    "confusion": "#a78bfa",
    "curiosity": "#60a5fa",
    "desire": "#f472b6",
    "disappointment": "#6b7280",
    "disapproval": "#ef4444",
    "disgust": "#dc2626",
    "embarrassment": "#f9a8d4",
    "excitement": "#facc15",
    "fear": "#7c3aed",
    "gratitude": "#4ade80",
    "grief": "#475569",
    "joy": "#fde047",
    "love": "#fb7185",
    "nervousness": "#c084fc",
    "neutral": "#9ca3af",
    "optimism": "#86efac",
    "pride": "#818cf8",
    "realization": "#7dd3fc",
    "relief": "#6ee7b7",
    "remorse": "#94a3b8",
    "sadness": "#64748b",
    "surprise": "#f0abfc",
    "disappointment": "#6b7280",
}


def render(df_review):
    apply_style()
    page_header("Emotion Analysis", "Analisis emosi dari review pengguna berdasarkan dataset Yelp")

    df = df_review.copy()

    if "top_emotion" not in df.columns:
        st.warning("Kolom top_emotion tidak tersedia di dataset.")
        return

    df["top_emotion"] = df["top_emotion"].fillna("neutral").str.lower().str.strip()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    section_label("Overview")

    total        = len(df)
    top_emotion  = df["top_emotion"].value_counts().idxmax()
    top_count    = df["top_emotion"].value_counts().max()
    unique_emo   = df["top_emotion"].nunique()
    neutral_pct  = (df["top_emotion"] == "neutral").sum() / total * 100

    kpi_style = "padding:16px;border-radius:10px;background:#1e2130;border:1px solid #2e3250;text-align:center;min-height:100px;display:flex;flex-direction:column;justify-content:center;"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Total Reviews</div>
            <div style="font-size:26px;font-weight:700;color:#60a5fa;">{total:,}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Top Emotion</div>
            <div style="font-size:26px;font-weight:700;color:#818cf8;">{top_emotion.title()}</div>
            <div style="font-size:12px;color:#6b7280;">{top_count:,} reviews</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Unique Emotions</div>
            <div style="font-size:26px;font-weight:700;color:#34d399;">{unique_emo}</div>
            <div style="font-size:12px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div style="{kpi_style}">
            <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Neutral Reviews</div>
            <div style="font-size:26px;font-weight:700;color:#9ca3af;">{neutral_pct:.1f}%</div>
            <div style="font-size:12px;color:#6b7280;">‎</div>
        </div>""", unsafe_allow_html=True)

    divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribution", "Trend", "By Rating", "vs Sentiment", "Review Table"])

    # ── Tab 1: Distribution ───────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            section_label("All Emotion Distribution — Treemap")
            counts = df["top_emotion"].value_counts().reset_index()
            counts.columns = ["emotion", "count"]
            counts["pct"] = (counts["count"] / total * 100).round(1)

            fig = px.treemap(
                counts,
                path=["emotion"],
                values="count",
                title=" ",
                color="emotion",
                color_discrete_map=EMOTION_COLORS,
                custom_data=["pct"],
            )
            fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[0]}%",
                hovertemplate="<b>%{label}</b><br>Reviews: %{value:,}<br>Percentage: %{customdata[0]}%<extra></extra>",
                textfont=dict(color="#e5e7eb"),
            )
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

        with c2:
            section_label("Emotion Distribution — Donut")
            top10 = counts.head(10)
            fig = go.Figure(go.Pie(
                labels=top10["emotion"].str.title(),
                values=top10["count"],
                hole=0.55,
                marker=dict(colors=[EMOTION_COLORS.get(e, "#818cf8") for e in top10["emotion"]]),
                textinfo="percent+label",
                textfont=dict(color="#e5e7eb"),
                hovertemplate="<b>%{label}</b><br>Reviews: %{value:,}<br>Percentage: %{percent}<extra></extra>",
            ))
            fig.update_layout(title="Top 10")
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    # ── Tab 2: Trend ──────────────────────────────────────────────────────────
    with tab2:
        if "date" in df.columns:
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"])
            d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
            d["year"]  = d["date"].dt.year  # ← agregasi tahunan untuk bump

            # Exclude neutral untuk analisis yang lebih bermakna
            d_no_neutral = d[d["top_emotion"] != "neutral"]

            top_emotions = d_no_neutral["top_emotion"].value_counts().head(6).index.tolist()
            d_trend = d_no_neutral[d_no_neutral["top_emotion"].isin(top_emotions)]

            # ── Bump Chart: per TAHUN (bukan bulan) ──────────────────────────
            section_label("Emotion Rank Per Year (Excluding Neutral)")
            trend_year = d_trend.groupby(["year", "top_emotion"]).size().reset_index(name="count")
            trend_year["rank"] = trend_year.groupby("year")["count"].rank(
                ascending=False, method="first"
            ).astype(int)

            fig_bump = go.Figure()
            for emotion in top_emotions:
                d_emo = trend_year[trend_year["top_emotion"] == emotion].sort_values("year")
                color = EMOTION_COLORS.get(emotion, "#818cf8")
                fig_bump.add_trace(go.Scatter(
                    x=d_emo["year"],
                    y=d_emo["rank"],
                    mode="lines+markers",
                    name=emotion.title(),
                    line=dict(color=color, width=2.5),
                    marker=dict(size=10, color=color),
                    hovertemplate=f"<b>{emotion.title()}</b><br>Year: %{{x}}<br>Rank: %{{y}}<br>Reviews: %{{customdata:,}}<extra></extra>",
                    customdata=d_emo["count"],
                ))
            fig_bump.update_layout(
                title=" ",
                yaxis=dict(autorange="reversed", tickvals=list(range(1, 7)),
                        title="Rank", gridcolor=GRID_COLOR),
                xaxis=dict(title="", dtick=1),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(style_fig(fig_bump, height=380), use_container_width=True)

            divider()

            # ── Emotion Composition Over Years (Stacked Area) ─────────────────
            section_label("Emotion Composition Over Years (Excluding Neutral)")
            trend_area = d_trend.groupby(["year", "top_emotion"]).size().reset_index(name="count")

            # Normalisasi ke persen per tahun
            total_per_year = trend_area.groupby("year")["count"].transform("sum")
            trend_area["pct"] = (trend_area["count"] / total_per_year * 100).round(1)

            fig_area = go.Figure()
            for emotion in top_emotions:
                d_emo = trend_area[trend_area["top_emotion"] == emotion].sort_values("year")
                color = EMOTION_COLORS.get(emotion, "#818cf8")
                fig_area.add_trace(go.Scatter(
                    x=d_emo["year"],
                    y=d_emo["pct"],
                    name=emotion.title(),
                    mode="lines",
                    stackgroup="one",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    hovertemplate=f"<b>{emotion.title()}</b><br>Year: %{{x}}<br>Percentage: %{{y:.1f}}%<br>Reviews: %{{customdata:,}}<extra></extra>",
                    customdata=d_emo["count"],
                ))

            fig_area.update_layout(
                title=" ",
                xaxis=dict(title="", dtick=1),
                yaxis=dict(title="Percentage (%)", range=[0, 100], gridcolor=GRID_COLOR),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(style_fig(fig_area, height=350), use_container_width=True)

    # ── Tab 3: By Rating ──────────────────────────────────────────────────────
    with tab3:
        section_label("Emotion Distribution per Star Rating")
        if "stars" in df.columns:
            top_emotions = df["top_emotion"].value_counts().head(8).index.tolist()
            d = df[df["top_emotion"].isin(top_emotions)].copy()
            d["top_emotion"] = d["top_emotion"].str.title()  # ← title case

            # Sesuaikan EMOTION_COLORS key jadi title case juga
            emotion_colors_title = {k.title(): v for k, v in EMOTION_COLORS.items()}

            pivot = d.groupby(["stars", "top_emotion"]).size().reset_index(name="count")
            total_per_star = pivot.groupby("stars")["count"].transform("sum")
            pivot["pct"] = (pivot["count"] / total_per_star * 100).round(1)

            fig = px.bar(pivot, x="stars", y="pct", color="top_emotion",
                        barmode="stack",
                        title="Top 8",
                        color_discrete_map=emotion_colors_title,
                        labels={"stars": "Star Rating", "pct": "Reviews", "top_emotion": ""})
            fig.update_traces(
                hovertemplate="<b>%{x}— %{fullData.name}</b><br>Percentage: %{y:.1f}%<extra></extra>"
            )
            fig.update_layout(
                xaxis=dict(tickvals=[1, 2, 3, 4, 5]),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(style_fig(fig, height=380), use_container_width=True)

            divider()
            section_label("Heatmap — Emotion vs Star Rating")
            pivot_hm = d.groupby(["top_emotion", "stars"]).size().reset_index(name="count")
            hm = pivot_hm.pivot_table(index="top_emotion", columns="stars", values="count", fill_value=0)
            fig2 = px.imshow(hm, color_continuous_scale="Purples",
                            title=" ",
                            labels={"x": "Stars", "y": "Emotion", "color": "Reviews"},
                            text_auto=True, aspect="auto")
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(fig2, height=380), use_container_width=True)
        else:
            st.info("Kolom stars tidak tersedia")

    # ── Tab 4: vs Sentiment ───────────────────────────────────────────────────
    with tab4:
        if "sentiment_score" in df.columns:
            emotion_colors_title = {k.title(): v for k, v in EMOTION_COLORS.items()}

            section_label("Avg Sentiment Score vs Review Volume per Emotion")
            avg = df.groupby("top_emotion").agg(
                avg_score=("sentiment_score", "mean"),
                count=("sentiment_score", "count")
            ).reset_index()
            avg["avg_score"] = avg["avg_score"].round(3)
            avg["top_emotion"] = avg["top_emotion"].str.title()
            avg["color"] = avg["top_emotion"].map(lambda e: emotion_colors_title.get(e, "#818cf8"))

            fig = go.Figure(go.Scatter(
                x=avg["avg_score"],
                y=avg["top_emotion"],
                mode="markers",
                marker=dict(
                    size=avg["count"],
                    sizemode="area",
                    sizeref=2. * avg["count"].max() / (40. ** 2),
                    sizemin=6,
                    color=avg["color"],
                    opacity=0.85,
                    line=dict(width=1, color="#1e2130"),
                ),
                text=avg["top_emotion"],
                customdata=list(zip(avg["avg_score"], avg["count"])),
                hovertemplate="<b>%{text}</b><br>Avg Score: %{customdata[0]:.3f}<br>Reviews: %{customdata[1]:,}<extra></extra>",
            ))
            fig.update_layout(
                title=" ",
                xaxis=dict(title="Avg Sentiment Score", range=[0, 1.05], gridcolor=GRID_COLOR),
                yaxis=dict(title="", categoryorder="mean ascending"),
            )
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

        else:
            st.info("Kolom sentiment_score tidak tersedia")

    # ── Tab 5: Review Table ───────────────────────────────────────────────────
    with tab5:
        section_label("Review Table")
        col1, col2, col3 = st.columns(3)
        with col1:
            emotion_opts = ["All"] + sorted(df["top_emotion"].unique().tolist())
            emo_filter = st.selectbox("Filter Emotion", emotion_opts)
        with col2:
            if "sentiment_label" in df.columns:
                sent_filter = st.selectbox("Filter Sentiment", ["All", "POSITIVE", "NEGATIVE"])
            else:
                sent_filter = "All"
        with col3:
            if "stars" in df.columns:
                star_opt = st.multiselect("Filter Stars", sorted(df["stars"].dropna().unique()),
                                        default=sorted(df["stars"].dropna().unique()))
            else:
                star_opt = None

        df_table = df.copy()
        if emo_filter != "All":
            df_table = df_table[df_table["top_emotion"] == emo_filter]
        if sent_filter != "All" and "sentiment_label" in df_table.columns:
            df_table = df_table[df_table["sentiment_label"] == sent_filter]
        if star_opt and "stars" in df_table.columns:
            df_table = df_table[df_table["stars"].isin(star_opt)]

        show_cols = [c for c in ["review_id", "stars", "date", "top_emotion",
                                  "sentiment_label", "sentiment_score", "text"] if c in df_table.columns]
        rename_cols = {
            "review_id": "Review ID",
            "stars": "Stars",
            "date": "Date",
            "top_emotion": "Emotion",
            "sentiment_label": "Sentiment",
            "sentiment_score": "Score",
            "text": "Text",
        }
        st.dataframe(df_table[show_cols].rename(columns=rename_cols).head(200).reset_index(drop=True),
                     use_container_width=True, height=380)