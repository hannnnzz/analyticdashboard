import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_business(path="BusinessData.xlsx"):
    df = pd.read_excel(path)
    for c in ["stars", "review_count", "is_open", "latitude", "longitude",
              "RestaurantsPriceRange2", "RestaurantsPriceRange2_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def split_categories(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if x]
    s = str(val)
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [p.strip().strip("\"'") for p in s.split(",") if p.strip()]


def explode_categories(df, col="categories"):
    if col not in df.columns:
        df["categories_exploded"] = np.nan
        return df
    tmp = df.copy()
    tmp["categories_exploded"] = tmp[col].apply(split_categories)
    tmp = tmp.explode("categories_exploded")
    tmp["categories_exploded"] = tmp["categories_exploded"].replace("", np.nan)
    return tmp