"""
Chart functions khusus Business Intelligence.
Semua chart pakai style dark theme dari components/style.py.

Aturan utama:
- Angka asli (count) dipakai di semua chart kecuali yang memang harus avg/persen
  (Avg Rating, Avg Review per category, distribusi proporsi, box plot).
- Semua bar chart vertikal.
- Semua "Top N" default = 10 (batas tampilan agar rapi).
- Heights distandarkan untuk layout dashboard.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.style import style_fig, COLOR_SEQ, CHART_BG, GRID_COLOR, FONT_COLOR

# ------------------------------------------------------------
# MARKET OVERVIEW
# ------------------------------------------------------------

def chart_rating_distribution(df):
    if "RestaurantsPriceRange2_num" not in df.columns:
        return None
    price_map = {1: "$ Budget", 2: "$$ Mid-Range", 3: "$$$ Upscale", 4: "$$$$ Fine Dining"}
    d = df["RestaurantsPriceRange2_num"].dropna().astype(int)
    counts = d.value_counts().sort_index().reset_index()
    counts.columns = ["tier", "count"]
    counts["label"] = counts["tier"].map(price_map)
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(1)

    fig = px.bar(counts, x="label", y="count",
                 title=" ",
                 color="tier", color_continuous_scale="Purples",
                 labels={"label": "", "count": "Businesses"},
                 text=counts["pct"].apply(lambda x: f"{x}%"),
                 hover_data={"count": True, "pct": True, "tier": False})
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return style_fig(fig, height=360)


def chart_top_cities(df, n=10):
    if "city" not in df.columns:
        return None
    d = df["city"].value_counts().nlargest(n).reset_index()
    d.columns = ["city", "count"]
    fig = px.bar(d, x="city", y="count",
                 title=f"Top {n} ",
                 color="count", color_continuous_scale="Blues",
                 labels={"count": "Businesses", "city": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=380)


def chart_open_closed(df):
    if "is_open" not in df.columns:
        return None
    oc = df.groupby("is_open").agg(n=("business_id", "nunique")).reset_index()
    oc["label"] = oc["is_open"].map({1: "Open", 0: "Closed"})
    fig = px.pie(oc, names="label", values="n", hole=0.55,
                 title="Business Operating Status",
                 color="label",
                 color_discrete_map={"Open": "#34d399", "Closed": "#f87171"})
    fig.update_traces(textposition="outside", textinfo="value+percent+label",
                      textfont_color="#e5e7eb")
    return style_fig(fig, height=360)


def chart_top_categories(df_exploded, n=10):
    if "categories_exploded" not in df_exploded.columns:
        return None
    top = (df_exploded.groupby("categories_exploded", dropna=True)
           .agg(cnt=("business_id", "nunique")).reset_index()
           .sort_values("cnt", ascending=False).head(n))
    return top  


def chart_business_map(df):
    if not {"latitude", "longitude"}.issubset(df.columns):
        return None
    d = df.dropna(subset=["latitude", "longitude"]).copy()
    if d.empty:
        return None
    hover = [c for c in ["name", "city", "state", "stars", "review_count"] if c in d.columns]
    fig = px.scatter_mapbox(
        d, lat="latitude", lon="longitude",
        color="stars" if "stars" in d.columns else None,
        size="review_count" if "review_count" in d.columns else None,
        size_max=18,
        hover_data=hover,
        color_continuous_scale="RdYlGn",
        zoom=3, height=440,
        title=" ",
    )
    # keep mapbox style & apply theme layout from style_fig
    fig.update_layout(mapbox_style="open-street-map",
                      margin=dict(l=0, r=0, t=40, b=0),
                      title_font=dict(size=14, color="#e5e7eb"))
    return style_fig(fig, height=440)


def chart_state_distribution(df, top=10):
    if "state" not in df.columns:
        return None
    d = df["state"].value_counts().head(top).reset_index()
    d.columns = ["state", "count"]
    fig = px.bar(d, x="state", y="count",
                 title=f"Top {top}",
                 color="count", color_continuous_scale="Blues",
                 labels={"state": "State", "count": "Businesses"})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=380)


def chart_review_count_dist(df):
    if "review_count" not in df.columns:
        return None
    d = df[df["review_count"] <= df["review_count"].quantile(0.95)].copy()
    fig = px.histogram(d, x="review_count", nbins=40,
                       title=" ",
                       labels={"review_count": "Number of Reviews", "count": "Businesses"},
                       color_discrete_sequence=[COLOR_SEQ[1]])
    fig.update_layout(xaxis=dict(title="Review Count"), yaxis=dict(title="Businesses"))
    return style_fig(fig, height=360)


# ------------------------------------------------------------
# REPUTATION & POPULARITY
# ------------------------------------------------------------

def chart_stars_vs_review(df):
    if not {"stars", "review_count"}.issubset(df.columns):
        return None
    d = (df.drop_duplicates(subset=["business_id"])
         .dropna(subset=["stars", "review_count"]))
    # limit for performance but keep representative sample
    if d.shape[0] > 5000:
        d = d.sample(5000, random_state=42)
    hover = [c for c in ["name", "city", "categories"] if c in d.columns]
    fig = px.scatter(d, x="review_count", y="stars",
                     opacity=0.65, hover_data=hover,
                     color="stars", color_continuous_scale="RdYlGn",
                     title=" ",
                     labels={"review_count": "Review Count", "stars": "Rating"})
    fig.update_coloraxes(showscale=False)
    return style_fig(fig, height=420)


def chart_avg_rating_per_category(df_exploded, min_n=5, top=10):
    # Avg rating per category (top = 10 default)
    if "categories_exploded" not in df_exploded.columns or "stars" not in df_exploded.columns:
        return None
    s = (df_exploded.groupby("categories_exploded", dropna=True)
         .agg(avg_stars=("stars", "mean"), cnt=("business_id", "nunique")).reset_index())
    s = s[s["cnt"] >= min_n].sort_values("avg_stars", ascending=False).head(top)
    fig = px.bar(s, x="categories_exploded", y="avg_stars",
                 color="avg_stars", color_continuous_scale="RdYlGn",
                 title=f"Top {top}, Min {min_n} Businesses",
                 labels={"avg_stars": "Avg Rating", "categories_exploded": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=420)


def chart_avg_review_per_category(df_exploded, min_n=5, top=10):
    # Avg review count per category (top = 10 default)
    if "categories_exploded" not in df_exploded.columns or "review_count" not in df_exploded.columns:
        return None
    s = (df_exploded.groupby("categories_exploded", dropna=True)
         .agg(avg_rev=("review_count", "mean"), cnt=("business_id", "nunique")).reset_index())
    s = s[s["cnt"] >= min_n].sort_values("avg_rev", ascending=False).head(top)
    fig = px.bar(s, x="categories_exploded", y="avg_rev",
                 color="avg_rev", color_continuous_scale="Blues",
                 title=f"Top {top}, Min {min_n} Businesses",
                 labels={"avg_rev": "Avg Reviews", "categories_exploded": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=420)


def chart_rating_vs_price(df):
    price_col = next((p for p in ["RestaurantsPriceRange2_num", "RestaurantsPriceRange2"]
                      if p in df.columns), None)
    if not price_col or "stars" not in df.columns:
        return None
    d = df.dropna(subset=["stars", price_col]).copy()
    d[price_col] = d[price_col].astype(str)
    label_map = {"1": "$ Budget", "2": "$$ Mid", "3": "$$$ Upscale", "4": "$$$$ Fine"}
    d[price_col] = d[price_col].map(lambda x: label_map.get(x, x))
    fig = px.box(d, x=price_col, y="stars",
                 color=price_col,
                 color_discrete_sequence=COLOR_SEQ,
                 title=" ",
                 labels={price_col: "Price Range", "stars": "Rating"})
    fig.update_traces(boxmean=True)
    return style_fig(fig, height=380)


def chart_top_businesses_by_reviews(df, n=10):
    if not {"name", "review_count", "stars"}.issubset(df.columns):
        return None
    d = (df.dropna(subset=["name", "review_count"])
         .drop_duplicates(subset=["business_id"])
         .nlargest(n, "review_count"))
    fig = px.bar(d, x="name", y="review_count",
                 color="stars", color_continuous_scale="RdYlGn",
                 title=f"Top {n}",
                 labels={"review_count": "Review Count", "name": "", "stars": "★"})
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=420)


def chart_rating_heatmap_city_category(df_exploded, top_cities=8, top_cats=8):
    if not {"city", "categories_exploded", "stars"}.issubset(df_exploded.columns):
        return None
    top_c = df_exploded["city"].value_counts().nlargest(top_cities).index
    top_k = (df_exploded.groupby("categories_exploded")["business_id"]
             .nunique().nlargest(top_cats).index)
    d = df_exploded[df_exploded["city"].isin(top_c) & df_exploded["categories_exploded"].isin(top_k)]
    if d.empty:
        return None
    pivot = d.groupby(["city", "categories_exploded"])["stars"].mean().unstack(fill_value=np.nan)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(tickfont=dict(color=FONT_COLOR)),
    ))
    fig.update_layout(
        title=" ",
        xaxis=dict(title="Category", tickangle=-35, color=FONT_COLOR, gridcolor=GRID_COLOR),
        yaxis=dict(title="City", color=FONT_COLOR, gridcolor=GRID_COLOR),
    )
    return style_fig(fig, height=380)


def chart_star_breakdown(df):
    if "stars" not in df.columns:
        return None
    breakdown = df["stars"].dropna().value_counts().sort_index()
    total = breakdown.sum()
    pct = (breakdown / total * 100).round(1)
    colors = ["#f87171", "#fb923c", "#fbbf24", "#facc15", "#a3e635",
              "#4ade80", "#34d399", "#2dd4bf", "#818cf8"]

    fig = go.Figure()
    for i, (star, count) in enumerate(breakdown.items()):
        fig.add_trace(go.Bar(
            name=f"{star}★",
            x=[count],
            y=["Ratings"],
            orientation="h",
            marker_color=colors[i % len(colors)],
            text=f"{star}★  {pct[star]}%",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{star}★</b><br>Businesses: {int(count):,}<br>Percentage: {pct[star]}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Rating Breakdown (Business Count)",
        showlegend=False,
        xaxis=dict(title="Number of Businesses", color=FONT_COLOR, gridcolor=GRID_COLOR),
        yaxis=dict(showticklabels=False),
        height=160,
    )
    return style_fig(fig, height=160)


# ------------------------------------------------------------
# DIFFERENTIATION & STRATEGY
# ------------------------------------------------------------

def _bool_count(df, cols, label_col="feature"):
    """
    Hitung jumlah bisnis (True) dan avg_stars per fitur boolean.
    Untuk kolom _norm: nilai selain no/none/false/0 dianggap True.
    Contoh: WiFi_norm=free -> True, no -> False.
    """
    FALSE_VALS = {"false", "0", "no", "none", "unknown", "nan", ""}
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        # treat missing as explicit False (so counts are correct)
        ser = df[c].fillna("").astype(str).str.strip().str.lower()
        if c.lower().endswith("_norm"):
            flag = ser.map(lambda x: 0 if x in FALSE_VALS else 1)
        else:
            flag = ser.map(lambda x: 1 if x in {"true", "1", "yes"} else 0)
        count = int(flag.sum())
        avg_stars = df.loc[flag[flag == 1].index, "stars"].mean() if "stars" in df.columns and count > 0 else np.nan
        rows.append({label_col: c, "count": count, "avg_stars": avg_stars})
    return pd.DataFrame(rows)


def chart_facility_coverage(df):
    cols = [c for c in df.columns if any(k in c.lower() for k in
            ["acceptscreditcard", "outdoor", "hastv", "wifi", "reserv",
             "goodforkids", "goodforgroups", "delivery", "takeout",
             "wheelchairaccessible", "drivethr", "dogsallowed"])][:20]
    if not cols:
        return None
    d = _bool_count(df, cols, "facility").sort_values("count", ascending=False)
    d["facility"] = d["facility"].str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True).str.title()
    d["facility"] = d["facility"].str.replace("_Norm", "", regex=False)
    
    # rename setelah semua transformasi
    rename_facility = {
        "Business Accepts Credit Cards": "Business ACC Credit Cards",
        "Restaurants Good For Groups": "Rest Good For Groups",
    }
    d["facility"] = d["facility"].replace(rename_facility)
    
    d = d.reset_index(drop=True)
    d.index += 1
    return d


def chart_ambience_profile(df, top=10):
    cols = [c for c in df.columns if "ambience_" in c.lower()]
    if not cols or "stars" not in df.columns:
        return None
    d = _bool_count(df, cols, "ambience").dropna()
    d["ambience"] = d["ambience"].str.replace("Ambience_", "", case=False).str.title()
    d = d.sort_values("count", ascending=False).head(top)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Business Count",
        x=d["ambience"], y=d["count"],
        marker_color=COLOR_SEQ[0], yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        name="Avg Rating",
        x=d["ambience"], y=d["avg_stars"],
        mode="lines+markers",
        marker=dict(color=COLOR_SEQ[2], size=8),
        line=dict(color=COLOR_SEQ[2], width=2),
        yaxis="y2"
    ))
    fig.update_layout(
        title=f" ",
        yaxis=dict(title="Business Count", color=FONT_COLOR, gridcolor=GRID_COLOR),
        yaxis2=dict(title="Avg Rating", overlaying="y", side="right", color=COLOR_SEQ[2], range=[3, 5]),
        legend=dict(orientation="h", y=1.12),
    )
    return style_fig(fig, height=420)


def chart_dietary_restrictions(df, top=10):
    cols = [c for c in df.columns if "dietaryrestrictions_" in c.lower()]
    if not cols:
        return None
    d = _bool_count(df, cols, "diet").sort_values("count", ascending=False).head(top)
    d["diet"] = d["diet"].str.replace("DietaryRestrictions_", "", case=False).str.title()
    fig = px.bar(d, x="diet", y="count",
                 color="count", color_continuous_scale="Greens",
                 title=f" ",
                 labels={"count": "Businesses", "diet": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=420)


def chart_goodformeal(df, top=10):
    cols = [c for c in df.columns if "goodformeal_" in c.lower()]
    if not cols:
        return None
    d = _bool_count(df, cols, "meal").dropna().sort_values("count", ascending=False).head(top)
    d["meal"] = d["meal"].str.replace("GoodForMeal_", "", case=False).str.title()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Business Count",
        x=d["meal"], y=d["count"],
        marker_color=COLOR_SEQ[0],
    ))
    fig.add_trace(go.Scatter(
        name="Avg Rating",
        x=d["meal"], y=d["avg_stars"],
        mode="lines+markers",
        marker=dict(color=COLOR_SEQ[2], size=8),
        line=dict(color=COLOR_SEQ[2], width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title=f" ",
        yaxis=dict(title="Business Count", color=FONT_COLOR, gridcolor=GRID_COLOR),
        yaxis2=dict(title="Avg Rating", overlaying="y", side="right", color=COLOR_SEQ[2], range=[3, 5]),
        legend=dict(orientation="h", y=1.12),
    )
    return style_fig(fig, height=420)


def chart_parking(df, top=10):
    cols = [c for c in df.columns if "businessparking_" in c.lower()]
    if not cols:
        return None
    d = _bool_count(df, cols, "parking").sort_values("count", ascending=False).head(top)
    d["parking"] = d["parking"].str.replace("BusinessParking_", "", case=False).str.title()
    fig = px.bar(d, x="parking", y="count",
                 color="count", color_continuous_scale="Blues",
                 title=f" ",
                 labels={"count": "Businesses", "parking": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=420)


def chart_music_types(df, top=10):
    cols = [c for c in df.columns if "music_" in c.lower()]
    if not cols:
        return None
    d = _bool_count(df, cols, "music").sort_values("count", ascending=False).head(top)
    d["music"] = d["music"].str.replace("Music_", "", case=False).str.replace("_", " ").str.title()
    fig = px.bar(d, x="music", y="count",
                 color="count", color_continuous_scale="Purples",
                 title=f" ",
                 labels={"count": "Businesses", "music": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return style_fig(fig, height=360)


def chart_bestnights(df):
    cols = [c for c in df.columns if "bestnights_" in c.lower()]
    if not cols:
        return None
    d = _bool_count(df, cols, "day").sort_values("count", ascending=False)
    d["day"] = d["day"].str.replace("BestNights_", "", case=False).str.title()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d["day"] = pd.Categorical(d["day"], categories=day_order, ordered=True)
    d = d.sort_values("day")
    fig = px.bar(d, x="day", y="count",
                 color="count", color_continuous_scale="Oranges",
                 title=" ",
                 labels={"count": "Businesses", "day": ""})
    fig.update_coloraxes(showscale=False)
    return style_fig(fig, height=420)


def chart_noise_wifi_alcohol(df):
    rows = []
    for col, label in [("NoiseLevel_norm", "Noise Level"),
                       ("WiFi_norm", "WiFi"),
                       ("Alcohol_norm", "Alcohol"),
                       ("Smoking_norm", "Smoking"),
                       ("RestaurantsAttire_norm", "Attire")]:
        if col in df.columns:
            vc = df[col].fillna("Unknown").astype(str).value_counts().reset_index()
            vc.columns = ["value", "count"]
            vc["attribute"] = label
            rows.append(vc)
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d["value"] = d["value"].str.replace("_", " ").str.title()

    totals = d.groupby("attribute")["count"].transform("sum")
    d["pct"] = (d["count"] / totals * 100).round(1)
    d["bar_label"] = d["pct"].apply(lambda x: f"{x}%" if x >= 8 else "")

    order = ["Noise Level", "WiFi", "Alcohol", "Smoking", "Attire"]
    d["attribute"] = pd.Categorical(d["attribute"], categories=[a for a in order if a in d["attribute"].unique()], ordered=True)

    fig = px.bar(d, y="attribute", x="pct", color="value",
                 orientation="h",
                 barmode="stack",
                 title=" ",
                 color_discrete_sequence=COLOR_SEQ,
                 labels={"pct": "%", "attribute": "", "value": ""},
                 custom_data=["value", "count", "pct"])
    fig.update_traces(
        text=d["bar_label"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=11, color="#0f1117"),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Businesses: %{customdata[1]:,}<br>Percentage: %{customdata[2]}%<extra></extra>"
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", y=-0.25),
    )
    return style_fig(fig, height=360)


def chart_operating_hours_heatmap(df):
    hour_cols = [c for c in df.columns if c.startswith("hours_")]
    if not hour_cols:
        return None

    def parse_open_hour(val):
        try:
            return int(str(val).split(":")[0])
        except Exception:
            return np.nan

    def parse_close_hour(val):
        try:
            parts = str(val).split("-")
            if len(parts) < 2:
                return np.nan
            return int(parts[1].split(":")[0])
        except Exception:
            return np.nan

    days, open_hrs, close_hrs = [], [], []
    for col in hour_cols:
        day = col.replace("hours_", "")
        opens  = df[col].apply(parse_open_hour).dropna()
        closes = df[col].apply(parse_close_hour).dropna()
        if len(opens) > 0 and len(closes) > 0:
            days.append(day)
            open_hrs.append(opens.mean())
            close_hrs.append(closes.mean())

    if not days:
        return None

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    summary = pd.DataFrame({"day": days, "avg_open": open_hrs, "avg_close": close_hrs})
    summary["day"] = pd.Categorical(summary["day"], categories=day_order, ordered=True)
    summary = summary.sort_values("day").dropna()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg Open Hour", x=summary["day"], y=summary["avg_open"],
                         marker_color=COLOR_SEQ[1]))
    fig.add_trace(go.Bar(name="Avg Close Hour", x=summary["day"], y=summary["avg_close"],
                         marker_color=COLOR_SEQ[0]))
    fig.update_layout(
        title=" ",
        barmode="group",
        yaxis=dict(title="Hour of Day (24h)", color=FONT_COLOR, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", y=1.12),
    )
    return style_fig(fig, height=380)