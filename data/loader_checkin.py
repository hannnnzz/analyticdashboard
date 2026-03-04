import pandas as pd
import streamlit as st


@st.cache_data
def load_checkin(path="CheckinData.xlsx"):
    """
    Kolom utama: _id, business_id, first_checkin_date, last_checkin_date,
        total_checkins, active_days_span, active_years, checkins_per_year,
        checkins_per_month, peak_hour, peak_day_of_week, peak_month,
        weekend_ratio, weekday_ratio, morning_ratio, afternoon_ratio,
        evening_ratio, night_ratio, is_churned, hazard_indicator,
        overdue_visit_flag, visit_frequency_label, covid_full_survival,
        post_covid_recovery_flag, pre_covid_checkins, covid_checkins,
        post_covid_checkins, checkin_trend_slope, yoy_growth, ...
    """
    df = pd.read_excel(path)
    for c in ["total_checkins", "active_days_span", "active_years",
              "checkins_per_year", "checkins_per_month", "weekend_ratio",
              "weekday_ratio", "hazard_indicator", "is_churned",
              "days_since_last_checkin", "avg_gap_days", "yoy_growth",
              "checkin_trend_slope", "pre_covid_checkins", "covid_checkins",
              "post_covid_checkins"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["first_checkin_date", "last_checkin_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df