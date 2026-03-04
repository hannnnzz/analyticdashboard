import streamlit as st
import numpy as np
from components.style import section_label


def _card(label, value, sub="", accent_class="kpi-accent"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {accent_class}">{value}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(df, df_exploded):
    section_label("Overview")
    cols = st.columns(5)

    with cols[0]:
        total = int(df["business_id"].nunique()) if "business_id" in df.columns else len(df)
        _card("Total Businesses", f"{total:,}", accent_class="kpi-blue")

    with cols[1]:
        avg = round(float(df["stars"].mean()), 2) if "stars" in df.columns else 0
        color = "kpi-green" if avg >= 4 else "kpi-yellow" if avg >= 3 else "kpi-red"
        _card("Avg Rating", f"{avg} ★", sub="‎", accent_class=color)

    with cols[2]:
        total_cities = int(df["city"].nunique()) if "city" in df.columns else 0
        total_states = int(df["state"].nunique()) if "state" in df.columns else 0
        _card("Cities", f"{total_cities:,}", sub=f"Across {total_states} States", accent_class="kpi-accent")

    with cols[3]:
        if "is_open" in df.columns:
            pct = df["is_open"].mean() * 100
            color = "kpi-green" if pct >= 70 else "kpi-yellow"
            _card("Open Rate", f"{pct:.1f}%", sub="‎",  accent_class=color)
        else:
            _card("Open Rate", "N/A")

    with cols[4]:
        if "review_count" in df.columns:
            total_rev = int(df["review_count"].sum())
            avg_rev   = round(df["review_count"].mean(), 1)
            _card("Total Reviews", f"{total_rev:,}", sub=f"avg {avg_rev}/business", accent_class="kpi-yellow")
        else:
            _card("Total Reviews", "N/A")


def render_kpi_row_secondary(df, df_exploded):
    cols = st.columns(4)

    with cols[0]:
        if "categories_exploded" in df_exploded.columns:
            top = (df_exploded.groupby("categories_exploded", dropna=True)
                   .agg(n=("business_id", "nunique")).reset_index()
                   .sort_values("n", ascending=False))
            if not top.empty:
                r = top.iloc[0]
                _card("Top Category", str(r["categories_exploded"]),
                      sub=f"{int(r['n']):,} businesses", accent_class="kpi-accent")
            else:
                _card("Top Category", "N/A")
        else:
            _card("Top Category", "N/A")

    with cols[1]:
        if "stars" in df.columns:
            top5 = df[df["stars"] >= 4.5]
            pct = len(top5) / len(df) * 100 if len(df) > 0 else 0
            _card("4.5+ Businesses", f"{len(top5):,}", sub=f"{pct:.1f}% of total", accent_class="kpi-green")
        else:
            _card("4.5+ Businesses", "N/A")

    with cols[2]:
        if "review_count" in df.columns:
            high_review = df[df["review_count"] >= 100]
            _card("High Review Business", f"{len(high_review):,}", sub="100+ reviews", accent_class="kpi-blue")
        else:
            _card("High Review Business", "N/A")

    with cols[3]:
        if "RestaurantsPriceRange2_num" in df.columns:
            avg_price = df["RestaurantsPriceRange2_num"].dropna().mean()
            label = ["Budget", "Mid-Range", "Upscale", "Fine Dining"]
            idx = min(int(round(avg_price)) - 1, 3) if not np.isnan(avg_price) else 1
            _card("Avg Price Range", "$" * (idx + 1), sub=label[idx], accent_class="kpi-yellow")
        else:
            _card("Avg Price Range", "N/A")