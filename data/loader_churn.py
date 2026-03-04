import pandas as pd
import streamlit as st

@st.cache_data
def load_churn():
    return pd.read_excel("UserUniqueReview.xlsx")