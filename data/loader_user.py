import pandas as pd
import streamlit as st


@st.cache_data
def load_user(
    path_unique="UserUniqueReview.xlsx",
    path_json="sampled-userdata.json",
):
    """
    UserUniqueReview kolom: user_id, total_reviews, avg_stars, total_useful,
        total_funny, total_cool, avg_sentiment_score, avg_review_length,
        days_since_last_review, days_since_first_review, votes_per_review, churn

    sampled-userdata kolom: _id, user_id, name, review_count, yelping_since,
        useful, funny, cool, elite, friends, fans, average_stars, compliment_*
    """
    df_unique = pd.read_excel(path_unique)

    try:
        df_json = pd.read_json(path_json)
    except Exception:
        df_json = pd.DataFrame()

    if not df_json.empty and "user_id" in df_unique.columns and "user_id" in df_json.columns:
        df = df_unique.merge(df_json, on="user_id", how="left", suffixes=("", "_json"))
    else:
        df = df_unique

    for c in ["total_reviews", "avg_stars", "avg_sentiment_score",
              "avg_review_length", "days_since_last_review",
              "days_since_first_review", "votes_per_review", "churn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df