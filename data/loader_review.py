import pandas as pd
import streamlit as st


@st.cache_data
def load_review(path="ReviewDataNew.xlsx"):
    """
    Kolom: _id, review_id, user_id, business_id, stars, useful, funny, cool,
           text, date, sentiment_label, sentiment_score, top_emotion,
           clean_summary_text, summary
    """
    df = pd.read_excel(path)
    for c in ["stars", "useful", "funny", "cool", "sentiment_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df