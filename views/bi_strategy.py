import streamlit as st
from components.style import apply_style, page_header, section_label, divider
from components.charts_bi import (
    chart_facility_coverage, chart_ambience_profile,
    chart_goodformeal, chart_parking,
    chart_music_types, chart_bestnights,
    chart_dietary_restrictions, chart_noise_wifi_alcohol,
    chart_operating_hours_heatmap,
)


def render(df, df_exploded):
    apply_style()
    page_header("Differentiation & Strategy", "Fasilitas, ambience, dan karakteristik operasional bisnis dalam Yelp")
    divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Facilities & Amenities", "Dining Profile", "Atmosphere", "Operations"])

    with tab1:
        section_label("Facility & Amenity Coverage")
        df_facility = chart_facility_coverage(df)
        if df_facility is not None and not df_facility.empty:
            cols = st.columns(4)
            for i, row in df_facility.iterrows():
                with cols[(i - 1) % 4]:
                    st.markdown(f"""
                    <div style="padding:14px 16px;margin-bottom:8px;border-radius:10px;
                                background:#1e2130;border:1px solid #2e3250;text-align:center;">
                        <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:6px;">
                            {row['facility']}
                        </div>
                        <div style="font-size:22px;font-weight:700;color:#818cf8;">
                            {int(row['count']):,}
                        </div>
                        <div style="font-size:11px;color:#6b7280;margin-top:4px;">businesses</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No data")

        divider()

        c1, c2 = st.columns(2)
        with c1:
            section_label("Parking Type Availability")
            fig = chart_parking(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")
        with c2:
            section_label("Dietary Restrictions Support")
            fig = chart_dietary_restrictions(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Good For Meal — Coverage vs Avg Rating")
            fig = chart_goodformeal(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")
        with c2:
            section_label("Best Nights for Business")
            fig = chart_bestnights(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with tab3:
        section_label("Ambience Type — Coverage vs Avg Rating")
        fig = chart_ambience_profile(df)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

        divider()

        c1, c2 = st.columns(2)
        with c1:
            section_label("Music Types Availability")
            fig = chart_music_types(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")
        with c2:
            section_label("Normalized Attributes Distribution")
            fig = chart_noise_wifi_alcohol(df)
            st.plotly_chart(fig, use_container_width=True) if fig else st.info("No data")

    with tab4:
        section_label("Average Operating Hours by Day of Week")
        fig = chart_operating_hours_heatmap(df)
        st.plotly_chart(fig, use_container_width=True) if fig else st.info("Kolom hours_* tidak tersedia")