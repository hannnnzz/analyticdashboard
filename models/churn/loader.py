"""
Loader untuk model Churn Prediction.

File yang dibutuhkan di folder models/churn/:
    - ChurnPredict.pkl
"""
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_churn_model(path="models/churn/ChurnPredict.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Model churn tidak bisa diload: {e}")
        return None


def predict_churn(model, df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """
    Predict churn probability untuk setiap user.
    Args:
        model: model yang sudah di-load
        df: DataFrame user
        feature_cols: list kolom fitur yang dipakai saat training
    Returns:
        pd.Series probability churn (0.0 – 1.0)
    TODO: sesuaikan feature_cols dengan yang dipakai saat training.
    """
    X = df[feature_cols].fillna(0)
    # return pd.Series(model.predict_proba(X)[:, 1], index=df.index)
    pass


# Fitur default yang dipakai model churn — sesuaikan jika berbeda
CHURN_FEATURES = [
    "total_reviews", "avg_stars", "avg_sentiment_score",
    "avg_review_length", "days_since_last_review",
    "days_since_first_review", "votes_per_review",
]