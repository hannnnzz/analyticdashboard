import pandas as pd
import streamlit as st

@st.cache_data
def load_wordcloud():
    return pd.read_excel("WordCloudData.xlsx")