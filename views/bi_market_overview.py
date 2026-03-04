import streamlit as st
import streamlit.components.v1 as components
from components.kpi import render_kpi_row, render_kpi_row_secondary
from components.style import apply_style, page_header, section_label, divider
from components.charts_bi import (
    chart_rating_distribution, chart_top_cities, chart_open_closed,
    chart_top_categories, chart_business_map, chart_state_distribution,
    chart_review_count_dist, chart_star_breakdown,
)


def render(df, df_exploded):
    apply_style()
    page_header("Market Overview", "Landscape bisnis secara keseluruhan dalam Yelp")

    render_kpi_row(df, df_exploded)
    st.markdown("<br>", unsafe_allow_html=True)
    render_kpi_row_secondary(df, df_exploded)
    divider()

    # ── Rating breakdown (full width, compact) ──────────────────────────────
    fig = chart_star_breakdown(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    divider()

    # ── Tab layout ───────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Geography", "Categories", "Volume"])

    with tab1:
        section_label("Business Location Map")
        fig = chart_business_map(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom latitude/longitude tidak tersedia")

        c1, c2 = st.columns(2)
        with c1:
            section_label("Business Count by State")
            fig = chart_state_distribution(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")
        with c2:
            section_label("Business Count by Cities")
            fig = chart_top_cities(df, n=15)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Categories by Business Count")
            top_cat = chart_top_categories(df_exploded, n=15)
            rename_map = {
                "Event Planning & Services": "Event Plan Services",
                "Arts & Entertainment": "Arts Entertain",
            }
            top_cat["categories_exploded"] = top_cat["categories_exploded"].replace(rename_map)
            if top_cat is not None and not top_cat.empty:
                max_cnt = top_cat["cnt"].max()
                rows = [top_cat.iloc[i:i+3] for i in range(0, len(top_cat), 3)]
                for chunk in rows:
                    cols3 = st.columns(3)
                    for j, (_, row) in enumerate(chunk.iterrows()):
                        pct = row["cnt"] / max_cnt * 100
                        with cols3[j]:
                            st.markdown(f"""
                            <div style="padding:12px 14px;margin-bottom:8px;border-radius:8px;
                                        background:#1e2130;border:1px solid #2e3250;text-align:center;">
                                <div style="font-size:12px;color:#9ca3af;margin-bottom:4px;">
                                    {row['categories_exploded']}
                                </div>
                                <div style="font-size:20px;font-weight:700;color:#818cf8;">
                                    {int(row['cnt']):,}
                                </div>
                                <div style="font-size:11px;color:#6b7280;">
                                    {pct:.1f}% of top
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No data")
        with c2:
            section_label("Open vs Closed")
            fig = chart_open_closed(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Price Range Distribution")
            fig = chart_rating_distribution(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")
        with c2:
            section_label("Review Count Distribution")
            fig = chart_review_count_dist(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")