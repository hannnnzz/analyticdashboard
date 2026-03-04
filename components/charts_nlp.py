import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# ── Import style dari components ─────────────────────────────────────────────
from components.style import style_fig, COLOR_SEQ, FONT_COLOR, GRID_COLOR, CHART_BG

SENTIMENT_COLORS = {"POSITIVE": "#34d399", "NEGATIVE": "#f87171", "NEUTRAL": "#fbbf24"}


# ── Donut distribusi sentiment ────────────────────────────────────────────────
def chart_sentiment_distribution(df):
    if "sentiment_label" not in df.columns:
        return None
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    colors = [SENTIMENT_COLORS.get(l, "#818cf8") for l in counts["label"]]
    fig = go.Figure(go.Pie(
        labels=counts["label"].str.title(),
        values=counts["count"],
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="percent+label",
        textfont=dict(color="#e5e7eb"),
        hovertemplate="<b>%{label}</b><br>Reviews: %{value:,}<br>Percentage: %{percent}<extra></extra>",
    ))
    fig.update_layout(title=" ")
    return style_fig(fig, height=320)


# ── Histogram sentiment score ─────────────────────────────────────────────────
def chart_sentiment_score_hist(df):
    if "sentiment_score" not in df.columns or "stars" not in df.columns:
        return None
    d = df.groupby("stars")["sentiment_score"].mean().reset_index()
    d.columns = ["stars", "avg_score"]
    d["avg_score"] = d["avg_score"].round(3)
    d["color"] = d["avg_score"].apply(lambda x: "#34d399" if x >= 0.7 else "#fbbf24" if x >= 0.5 else "#f87171")

    fig = px.bar(d, x="stars", y="avg_score",
                 title=" ",
                 labels={"stars": "Star Rating", "avg_score": "Avg Score"},
                 color="avg_score", color_continuous_scale="RdYlGn",
                 text=d["avg_score"].apply(lambda x: f"{x:.3f}"),
                 hover_data={"stars": True, "avg_score": True})
    fig.update_traces(textposition="outside", textfont=dict(size=11, color="#e5e7eb"))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(xaxis=dict(tickvals=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]),
                      yaxis=dict(range=[0, 1.1]))
    return style_fig(fig, height=320)


# ── Trend sentiment over time ─────────────────────────────────────────────────
def chart_sentiment_over_time(df):
    if "sentiment_label" not in df.columns or "date" not in df.columns:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    trend = (d.groupby(["month", "sentiment_label"])
              .size().reset_index(name="count"))
    total = trend.groupby("month")["count"].transform("sum")
    trend["pct"] = (trend["count"] / total * 100).round(1)
    fig = px.line(trend, x="month", y="pct", color="sentiment_label",
                  title="%",
                  color_discrete_map=SENTIMENT_COLORS,
                  labels={"month": "Bulan", "pct": "Reviews", "sentiment_label": ""},
                  markers=True)
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    return style_fig(fig, height=340)


# ── Sentiment per bintang ─────────────────────────────────────────────────────
def chart_stars_vs_sentiment(df):
    if "sentiment_label" not in df.columns or "stars" not in df.columns:
        return None
    d = df.groupby(["stars", "sentiment_label"]).size().reset_index(name="count")
    total = d.groupby("stars")["count"].transform("sum")
    d["pct"] = (d["count"] / total * 100).round(1)
    d["label"] = d["pct"].apply(lambda x: f"{x}%" if x >= 8 else "")

    fig = px.bar(d, x="stars", y="pct", color="sentiment_label",
                 barmode="stack",
                 title=" ",
                 color_discrete_map=SENTIMENT_COLORS,
                 labels={"stars": "Star Rating", "pct": "Reviews", "sentiment_label": ""})
    
    # set text & hovertemplate manual per trace
    for trace in fig.data:
        mask = d["sentiment_label"] == trace.name
        trace.text = d[mask]["label"].values
        trace.textposition = "inside"
        trace.insidetextanchor = "middle"
        trace.textfont = dict(size=11)
        trace.hovertemplate = "<b>%{x} — " + trace.name + "</b><br>Percentage: %{y:.1f}%<extra></extra>"

    fig.update_layout(
        xaxis=dict(tickvals=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]),
        legend=dict(orientation="h", y=1.1),
    )
    return style_fig(fig, height=340)


# ── Word frequency (pengganti word cloud) ─────────────────────────────────────
def chart_word_frequency(n=100):
    try:
        from wordcloud import WordCloud
        import matplotlib.colors as mcolors
        import re as _re
    except ImportError:
        return None

    from data.loader_wordcloud import load_wordcloud
    df_wc = load_wordcloud()

    if "cleaned_text" not in df_wc.columns:
        return None

    text = " ".join(df_wc["cleaned_text"].dropna().astype(str).tolist())
    if not text.strip():
        return None

    colors = ["#41295D", "#2F0743", "#800080", "#B06BCC", "#E0BBE4",
              "#957DAD", "#6C5B7B", "#C06C84", "#F67280", "#FF8C94"]
    colormap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

    wc = WordCloud(
        width=1200, height=500,
        background_color="white",
        colormap=colormap,
        collocations=True,
        collocation_threshold=3,
        max_words=n,
    ).generate(text)

    x, y, sizes, labels, colors_list = [], [], [], [], []
    for (word, freq), font_size, position, _, color_str in wc.layout_:
        labels.append(word)
        sizes.append(font_size)
        x.append(position[1])
        y.append(-position[0])
        match = _re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
        if match:
            r, g, b = map(int, match.groups())
            colors_list.append('#%02x%02x%02x' % (r, g, b))
        else:
            colors_list.append(mcolors.to_hex(color_str))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="text",
        text=labels,
        textfont=dict(size=sizes, color=colors_list),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        title="Word Cloud — Review Text",
        xaxis=dict(visible=False, range=[min(x)-10, max(x)+10]),
        yaxis=dict(visible=False, range=[min(y)-10, max(y)+10]),
        plot_bgcolor="white",
        paper_bgcolor=CHART_BG,
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
    )
    return fig