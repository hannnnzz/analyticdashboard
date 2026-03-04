"""
Loader untuk Business Summary.

File yang dibutuhkan di folder models/summary/:
    - yelp_business_summaries.pkl
"""
import pickle
import pandas as pd
import streamlit as st


@st.cache_data
def load_summaries(path="models/summary/yelp_business_summaries.pkl"):
    """
    Load business summaries.
    Output: DataFrame dengan kolom [business_id, summary]
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Sudah DataFrame, langsung return
    if isinstance(data, pd.DataFrame):
        return data

    # Fallback: jika dict {business_id: summary_text}
    if isinstance(data, dict):
        return pd.DataFrame(list(data.items()), columns=["business_id", "summary"])

    raise ValueError(f"Format pkl tidak dikenali: {type(data)}")