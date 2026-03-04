import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Analytics Dashboard", page_icon"💡", layout="wide", initial_sidebar_state="expanded")

from data.loader_business import load_business, explode_categories
from data.loader_review import load_review
from data.loader_user import load_user
from data.loader_checkin import load_checkin
from data.loader_tip import load_tip
from data.loader_edges import load_edges
from data.loader_churn import load_churn
from data.loader_wordcloud import load_wordcloud
from components.sidebar import render_sidebar
from models.summary.loader import load_summaries
from models.churn.loader import load_churn_model
from models.clustering.loader import load_user_segmented, load_kmeans
from models.recsys.loader import (
    load_ncf_model, load_label_encoder_user, load_label_encoder_biz,
    load_biz_features, load_biz_info, load_df_slim
)


# ── Load semua data ──────────────────────────────────────────────────────────
df_business  = load_business()
df_exploded  = explode_categories(df_business)
df_review    = load_review()
df_user      = load_user()
df_checkin   = load_checkin()
df_tip       = load_tip()
df_summary   = load_summaries()
df_edges     = load_edges()
df_churn     = load_churn()
df_wordcloud = load_wordcloud()
df_user_segmented = load_user_segmented()
kmeans_model      = load_kmeans()
churn_model       = load_churn_model()
ncf_model         = load_ncf_model()
le_user           = load_label_encoder_user()
le_biz            = load_label_encoder_biz()
biz_features      = load_biz_features()
biz_info          = load_biz_info()
df_slim           = load_df_slim()



# ── Inisialisasi page DULU sebelum sidebar ───────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# ── Sidebar filter ───────────────────────────────────────────────────────────
df_business, df_exploded, df_review, df_user, df_checkin, df_wordcloud = render_sidebar(
    df_business, df_exploded, df_review, df_user, df_checkin, df_wordcloud
)

# ── Navigation ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

SECTIONS = {
    "": ["Home"],
    
    "Business Intelligence": [
        "Market Overview",
        "Reputation & Popularity",
        "Differentiation & Strategy",
    ],
    "NLP": [
        "Sentiment Analysis",
        "Emotion Analysis",
        "Business Summary",
    ],
    "User Segmentation": [
        "Clustering, Social, Prediction",
    ],
    "Churn Analysis": [
        "User Overview",
        "Checkin Behavior",
    ],
    "Recommendation System": [
        "Business Recommendation for User",
    ],
}

for section, pages in SECTIONS.items():
    st.sidebar.markdown(f"**{section}**")
    for p in pages:
        if st.sidebar.button(p, key=p, use_container_width=True):
            st.session_state["page"] = p
            st.rerun()

page = st.session_state["page"]

# ── Route ────────────────────────────────────────────────────────────────────
if page == "Home":
    from views.home import render
    render()

elif page == "Market Overview":
    from views.bi_market_overview import render
    render(df_business, df_exploded)

elif page == "Reputation & Popularity":
    from views.bi_reputation import render
    render(df_business, df_exploded)

elif page == "Differentiation & Strategy":
    from views.bi_strategy import render
    render(df_business, df_exploded)

elif page == "Sentiment Analysis":
    from views.nlp_sentiment import render
    render(df_review)

elif page == "Emotion Analysis":
    from views.nlp_emotion import render
    render(df_review)

elif page == "Business Summary":
    from views.nlp_summary import render
    render(df_business, df_review, df_summary)

elif page == "Clustering, Social, Prediction":
    from views.clustering import render
    render(df_user_segmented, kmeans_model=kmeans_model, scaler=None, df_edges=df_edges)

elif page == "User Overview":
    from views.churn_user import render
    render(df_churn, df_user=df_user, churn_model=churn_model)

elif page == "Checkin Behavior":
    from views.churn_checkin import render
    render(df_checkin)

elif page == "Business Recommendation for User":
    from views.recsys import render
    render(
        ncf_model    = ncf_model,
        le_user      = le_user,
        le_biz       = le_biz,
        biz_features = biz_features,
        biz_info     = biz_info,
        df_slim      = df_slim,
        df_user      = df_user,
        df_business  = df_business,
    )

