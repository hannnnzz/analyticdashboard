import streamlit as st
from components.style import apply_style, page_header, section_label, divider
from components.charts_bi import (
    chart_stars_vs_review, chart_avg_rating_per_category,
    chart_avg_review_per_category, chart_rating_vs_price,
    chart_top_businesses_by_reviews, chart_rating_heatmap_city_category,
)


def render(df, df_exploded):
    apply_style()
    page_header("Reputation & Popularity", "Analisis rating, review, dan popularitas bisnis dalam Yelp")
    divider()

    # ── Row 1: scatter + top businesses ──────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        section_label("Stars vs Review Count")
        fig = chart_stars_vs_review(df)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with c2:
        section_label("Most Reviewed Businesses")
        fig = chart_top_businesses_by_reviews(df, n=15)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    divider()

    # ── Rating by price range (full width) ───────────────────────────────────
    section_label("Rating Distribution by Price Range")
    fig = chart_rating_vs_price(df)
    st.plotly_chart(fig, use_container_width=True) if fig else st.info("Kolom price range tidak tersedia")

    divider()

    # ── Heatmap city x category ───────────────────────────────────────────────
    section_label("Avg Rating — Top Cities × Top Categories")
    fig = chart_rating_heatmap_city_category(df_exploded)
    st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    divider()

    # ── Avg rating & review per category ─────────────────────────────────────
    min_biz = st.slider("Min businesses per category", 2, 20, 5, key="rep_min")

    c3, c4 = st.columns(2)
    with c3:
        section_label("Avg Rating per Category")
        fig = chart_avg_rating_per_category(df_exploded, min_n=min_biz)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with c4:
        section_label("Avg Review Count per Category")
        fig = chart_avg_review_per_category(df_exploded, min_n=min_biz)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")