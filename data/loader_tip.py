import pandas as pd
import streamlit as st


@st.cache_data
def load_tip(path="sampled_tipdata.json"):
    """
    Kolom: _id, user_id, business_id, text, date, compliment_count
    """
    try:
        df = pd.read_json(path)
    except Exception:
        df = pd.DataFrame(columns=["user_id", "business_id", "text", "date", "compliment_count"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "compliment_count" in df.columns:
        df["compliment_count"] = pd.to_numeric(df["compliment_count"], errors="coerce")
    return df