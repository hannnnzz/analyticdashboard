"""
Loader untuk model Clustering / User Segmentation.

File yang dibutuhkan di folder models/clustering/:
    - kmeans_model.pkl
    - hdbscan_model.pkl
    - scaler.pkl
    - segmentation_features.pkl
    - cluster_mapping.pkl
    - user_segmented.csv
"""
import joblib
import pandas as pd
import streamlit as st



@st.cache_resource
def load_kmeans(path="models/clustering/kmeans_model.pkl"):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path="models/clustering/scaler.pkl"):
    return joblib.load(path)


@st.cache_data
def load_segmentation_features(path="models/clustering/segmentation_features.pkl"):
    """
    Load list nama fitur yang dipakai saat training clustering.
    Ekspektasi output: list of str
    """
    with open(path, "rb") as f:
        return joblib.load(f)
    

@st.cache_data
def load_cluster_mapping(path="models/clustering/cluster_mapping.pkl"):
    """
    Load mapping cluster id → label/nama segment.
    Ekspektasi output: dict {cluster_id: label_str}
    TODO: sesuaikan jika struktur pkl berbeda.
    """
    with open(path, "rb") as f:
        return joblib.load(f)


@st.cache_data
def load_user_segmented(path="models/clustering/user_segmented.csv"):
    """
    Load hasil segmentasi user yang sudah di-assign cluster.
    Ekspektasi output: DataFrame dengan kolom [user_id, cluster_label, ...]
    """
    return pd.read_csv(path)