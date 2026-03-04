"""
Loader untuk model NLP Sentimo.

File yang dibutuhkan di folder models/sentimo/:
    - ReviewSentimentData.pkl
    - ReviewEmotionData.pkl
"""
import pickle
import streamlit as st


@st.cache_data
def load_sentiment_data(path="models/sentimo/ReviewSentimentData.pkl"):
    """
    Load hasil prediksi sentiment yang sudah diproses sebelumnya.
    Ekspektasi output: DataFrame dengan kolom minimal
        [review_id/user_id/business_id, sentiment_label, sentiment_score]
    TODO: sesuaikan parsing output jika struktur pkl berbeda.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


@st.cache_data
def load_emotion_data(path="models/sentimo/ReviewEmotionData.pkl"):
    """
    Load hasil prediksi emotion yang sudah diproses sebelumnya.
    Ekspektasi output: DataFrame dengan kolom minimal
        [review_id/user_id/business_id, top_emotion]
    TODO: sesuaikan parsing output jika struktur pkl berbeda.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data