import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.style import apply_style, page_header, section_label, divider, style_fig, FONT_COLOR, GRID_COLOR

SEGMENT_COLORS = {
    "Highly Engaged Influencers": "#818cf8",
    "Active Contributors":        "#34d399",
    "Regular Reviewers":          "#60a5fa",
    "Casual Users":               "#9ca3af",
}

# Sesuai cluster_mapping.pkl
CLUSTER_MAPPING = {3: "Highly Engaged Influencers", 0: "Active Contributors", 2: "Regular Reviewers", 1: "Casual Users"}

# Sesuai segmentation_features.pkl
FEATURES = [
    "review_count", "reviews_per_year", "average_stars", "rating_deviation",
    "account_age_days", "votes_per_review", "compliment_per_review",
    "fans", "network_size", "elite_flag", "elite_consistency",
    "engagement_score", "influence_score"
]

FEATURE_LABELS = {
    "review_count":          "Review Count",
    "reviews_per_year":      "Reviews / Year",
    "average_stars":         "Avg Stars",
    "rating_deviation":      "Rating Deviation",
    "account_age_days":      "Account Age (Days)",
    "account_age_years":     "Account Age (Years)",
    "votes_per_review":      "Votes / Review",
    "compliment_per_review": "Compliments / Review",
    "fans":                  "Fans",
    "network_size":          "Network Size",
    "elite_flag":            "Elite",
    "elite_consistency":     "Elite Consistency",
    "engagement_score":      "Engagement Score",
    "influence_score":       "Influence Score",
}

FEATURE_DESCRIPTIONS = {
    "review_count":          "Total number of reviews written by the user",
    "reviews_per_year":      "Average number of reviews the user writes per year",
    "average_stars":         "Average star rating given by the user across all reviews (1–5)",
    "rating_deviation":      "How inconsistent the user's ratings are (higher = more varied)",
    "account_age_days":      "How many days since the user first created their account",
    "votes_per_review":      "Average number of useful/funny/cool votes received per review",
    "compliment_per_review": "Average number of compliments received per review",
    "fans":                  "Total number of fans (followers) the user has",
    "network_size":          "Total number of friends in the user's network",
    "elite_flag":            "Whether the user has ever received Yelp Elite status (1 = Yes, 0 = No)",
    "elite_consistency":     "Proportion of account years in which the user held Elite status (0–1)",
    "engagement_score":      "Composite score reflecting how actively the user interacts on the platform",
    "influence_score":       "Composite score reflecting the user's overall influence and reach",
}


def render(df_user, kmeans_model=None, scaler=None, df_edges=None):
    apply_style()
    page_header("Clustering & Social Graph", "Segmentasi pengguna dalam Yelp menggunakan HDBSCAN dengan KMeans pipeline")

    df = df_user.copy()

    if "account_age_days" in df.columns:
        df["account_age_years"] = (df["account_age_days"] / 365).round(1)

    if "segment_label" not in df.columns or "KMeans_Cluster" not in df.columns:
        st.warning("Kolom clustering tidak lengkap di dataset.")
        return

    df["segment_label"] = df["segment_label"].fillna("Unknown")
    avail_features = [f for f in FEATURES if f in df.columns]

    FEATURES_DISPLAY = [f for f in [
        "review_count", "reviews_per_year", "average_stars",
        "fans", "network_size", "votes_per_review",
        "engagement_score", "influence_score", "elite_flag",
        "account_age_years",
    ] if f in df.columns]

    kpi_style = "padding:16px;border-radius:10px;background:#1e2130;border:1px solid #2e3250;text-align:center;min-height:120px;display:flex;flex-direction:column;justify-content:center;"

    # ── KPI ───────────────────────────────────────────────────────────────
    section_label("Overview")
    total      = len(df)
    n_segments = df["segment_label"].nunique()
    n_outliers = int(df["is_outlier"].sum()) if "is_outlier" in df.columns else None
    elite_pct  = df["elite_flag"].mean() * 100 if "elite_flag" in df.columns else None
    top_seg    = df["segment_label"].value_counts().idxmax()

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val, color in [
        (k1, "Total Users",     f"{total:,}",                                         "#60a5fa"),
        (k2, "Segments",        str(n_segments),                                      "#818cf8"),
        (k3, "Largest Segment", top_seg,                                              "#34d399"),
        (k4, "Outlier Users",   f"{n_outliers:,}" if n_outliers is not None else "-", "#f87171"),
        (k5, "Elite Users",     f"{elite_pct:.1f}%" if elite_pct is not None else "-","#facc15"),
        
    ]:
        with col:
            fs = "18px" if len(str(val)) > 10 else "24px"
            st.markdown(f"""<div style="{kpi_style}">
                <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
                <div style="font-size:{fs};font-weight:700;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Segment Overview", "Segment Profiling", "Influence & Network",
        "Elite Analysis", "User Lookup & Predict"
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 — SEGMENT OVERVIEW
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Segment Distribution")
            seg_counts = df["segment_label"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig = go.Figure(go.Pie(
                labels=seg_counts["segment"], values=seg_counts["count"], hole=0.55,
                marker=dict(colors=[SEGMENT_COLORS.get(s, "#9ca3af") for s in seg_counts["segment"]]),
                textinfo="percent+label", textfont=dict(color="#e5e7eb"),
                hovertemplate="<b>%{label}</b><br>Users: %{value:,}<br>Share: %{percent}<extra></extra>",
            ))
            fig.update_layout(title=" ", showlegend=False)
            st.plotly_chart(style_fig(fig, height=380), use_container_width=True)

        with c2:
            section_label("Number of Users per KMeans Cluster")
            km_counts = df["KMeans_Cluster"].value_counts().reset_index()
            km_counts.columns = ["cluster_id", "count"]
            km_counts["label"] = km_counts["cluster_id"].map(CLUSTER_MAPPING).fillna(km_counts["cluster_id"].astype(str))
            km_counts["color"] = km_counts["label"].map(lambda s: SEGMENT_COLORS.get(s, "#818cf8"))
            km_counts = km_counts.sort_values("cluster_id")
            fig2 = go.Figure(go.Bar(
                x=km_counts["count"], y=km_counts["label"], orientation="h",
                marker_color=km_counts["color"],
                text=km_counts["count"].apply(lambda x: f"{x:,}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Users: %{x:,}<extra></extra>",
            ))
            fig2.update_layout(title=" ",
                               xaxis=dict(title="Users", gridcolor=GRID_COLOR),
                               yaxis=dict(title=""), showlegend=False)
            st.plotly_chart(style_fig(fig2, height=380), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 — SEGMENT PROFILING
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        if not avail_features:
            st.info("Fitur segmentasi tidak tersedia.")
        else:
            section_label("Radar Chart — Profil per Segment")
            avg_seg = df.groupby("segment_label")[FEATURES_DISPLAY].mean()
            norm = (avg_seg - avg_seg.min()) / (avg_seg.max() - avg_seg.min() + 1e-9)
            cats = [FEATURE_LABELS.get(f, f) for f in FEATURES_DISPLAY]
            fig = go.Figure()
            for seg in norm.index:
                vals = norm.loc[seg].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill="toself", name=seg,
                    line=dict(color=SEGMENT_COLORS.get(seg, "#818cf8")),
                    fillcolor=SEGMENT_COLORS.get(seg, "#818cf8"),
                    opacity=0.25,
                    hovertemplate=f"<b>{seg}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
                ))
            fig.update_layout(
                title=" ",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR),
                    angularaxis=dict(gridcolor=GRID_COLOR),
                ),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

            divider()

            section_label("Avg Feature per Segment")
            feat_sel = st.selectbox(
                "Pilih Fitur", FEATURES_DISPLAY,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="feat_bar"
            )
            avg_feat = df.groupby("segment_label")[feat_sel].mean().reset_index()
            avg_feat.columns = ["segment", "avg"]
            avg_feat["color"] = avg_feat["segment"].map(lambda s: SEGMENT_COLORS.get(s, "#818cf8"))
            avg_feat = avg_feat.sort_values("avg", ascending=True)
            fig2 = go.Figure(go.Bar(
                x=avg_feat["avg"], y=avg_feat["segment"], orientation="h",
                marker_color=avg_feat["color"],
                text=avg_feat["avg"].round(2), textposition="outside",
                hovertemplate="<b>%{y}</b><br>Avg: %{x:.2f}<extra></extra>",
            ))
            fig2.update_layout(
                title=f" ",
                xaxis=dict(title=" ", gridcolor=GRID_COLOR),
                yaxis=dict(title=" "), showlegend=False,
            )
            st.plotly_chart(style_fig(fig2, height=440), use_container_width=True)

            divider()
            if "average_stars" in df.columns:
                section_label("Avg Star per Segment")
                avg_stars = df.groupby("segment_label")["average_stars"].mean().reset_index()
                avg_stars.columns = ["segment", "avg_stars"]
                avg_stars["color"] = avg_stars["segment"].map(lambda s: SEGMENT_COLORS.get(s, "#818cf8"))
                avg_stars = avg_stars.sort_values("avg_stars", ascending=True)

                fig3 = go.Figure(go.Bar(
                    x=avg_stars["avg_stars"], y=avg_stars["segment"], orientation="h",
                    marker_color=avg_stars["color"],
                    text=avg_stars["avg_stars"].apply(lambda x: f"{x:.2f}"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Avg Stars: %{x:.2f}<extra></extra>",
                ))
                fig3.update_layout(
                    title=" ",
                    xaxis=dict(title="", range=[0, 5.5], gridcolor=GRID_COLOR),
                    yaxis=dict(title=""),
                    showlegend=False,
                )
                st.plotly_chart(style_fig(fig3, height=300), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3 — INFLUENCE & NETWORK
    # ═══════════════════════════════════════════════════════════════════
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Network Size vs Influence Score")
            if "network_size" in df.columns and "influence_score" in df.columns:
                sample = df.sample(min(3000, len(df)), random_state=42)
                fig = px.scatter(sample, x="network_size", y="influence_score",
                                 color="segment_label", color_discrete_map=SEGMENT_COLORS,
                                 title=" ",
                                 labels={"network_size": "Network Size",
                                         "influence_score": "Influence Score", "segment_label": ""},
                                 opacity=0.7,
                                 hover_data=["name"] if "name" in df.columns else None)
                fig.update_traces(marker=dict(size=6))
                fig.update_layout(legend=dict(orientation="h", y=1.1))
                st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
            else:
                st.info("Kolom network_size atau influence_score tidak tersedia.")

        with c2:
            section_label("Biggest Segment")
            seg_counts = df["segment_label"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            seg_counts["pct"] = (seg_counts["count"] / total * 100).round(1)
            seg_counts["color"] = seg_counts["segment"].map(lambda s: SEGMENT_COLORS.get(s, "#818cf8"))

            fig = go.Figure(go.Bar(
                x=seg_counts["segment"],
                y=seg_counts["count"],
                marker_color=seg_counts["color"],
                text=seg_counts["pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Users: %{y:,}<extra></extra>",
            ))
            fig.update_layout(
                title=" ",
                xaxis=dict(title="", tickangle=-15),
                yaxis=dict(title="Users", gridcolor=GRID_COLOR),
                showlegend=False,
            )
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

        divider()
        section_label("User Connection Network by Influence Score")
        if "influence_score" in df.columns:
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                seg_filter = st.selectbox(
                    "Filter Segment", ["All"] + sorted(df["segment_label"].unique().tolist()),
                    key="inf_seg"
                )
            with col_f2:
                top_n = st.slider("Top N Users", min_value=10, max_value=100, value=30, step=10)

            df_inf    = df if seg_filter == "All" else df[df["segment_label"] == seg_filter]
            top_users = df_inf.nlargest(top_n, "influence_score")["user_id"].tolist()

            # Load edges & filter hanya user yang masuk top N
            try:
                if df_edges is None:
                    st.warning("Data edges tidak tersedia.")
                else:
                    df_edges_filtered = df_edges[
                        df_edges["source"].isin(top_users) & df_edges["target"].isin(top_users)
                    ]
                    
                df_edges = df_edges[
                    df_edges["source"].isin(top_users) & df_edges["target"].isin(top_users)
                ]

                if df_edges.empty:
                    st.info("Tidak ada koneksi antar top users di segment ini.")
                else:
                    import networkx as nx

                    G = nx.from_pandas_edgelist(df_edges, source="source", target="target")

                    # Layout
                    pos = nx.spring_layout(G, seed=42, k=1.5)

                    # Edges
                    edge_x, edge_y = [], []
                    for u, v in G.edges():
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]

                    # Nodes — merge dengan df untuk dapat segment & nama
                    node_meta = df_inf[df_inf["user_id"].isin(G.nodes())].set_index("user_id")
                    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        if node in node_meta.index:
                            row     = node_meta.loc[node]
                            seg     = row.get("segment_label", "Unknown")
                            name    = row.get("name", node)
                            inf     = row.get("influence_score", 0)
                            degree  = G.degree(node)
                            node_color.append(SEGMENT_COLORS.get(seg, "#9ca3af"))
                            node_size.append(8 + degree * 2)
                            node_text.append(f"<b>{name}</b><br>Segment: {seg}<br>Influence: {inf:.2f}<br>Connections: {degree}")
                        else:
                            node_color.append("#475569")
                            node_size.append(8)
                            node_text.append(node)

                    fig = go.Figure()

                    # Edge trace
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=0.8, color="#2e3250"),
                        hoverinfo="none", showlegend=False,
                    ))

                    # Node trace
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y, mode="markers",
                        marker=dict(size=node_size, color=node_color,
                                    line=dict(width=1, color="#1e2130")),
                        text=node_text,
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                    ))

                    # Legend manual per segment
                    for seg, color in SEGMENT_COLORS.items():
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(size=10, color=color),
                            name=seg, showlegend=True,
                        ))

                    fig.update_layout(
                        title=f"",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        legend=dict(orientation="h", y=1.1),
                        plot_bgcolor="#0e1117",
                    )
                    st.plotly_chart(style_fig(fig, height=500), use_container_width=True)

                    st.caption(f"Showing {G.number_of_nodes()} users and {G.number_of_edges()} connection. Node size = total connection.")

            except FileNotFoundError:
                st.warning("File UserEdges.xlsx tidak ditemukan.")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4 — ELITE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    with tab4:
        if "elite_flag" not in df.columns:
            st.info("Kolom elite_flag tidak tersedia.")
        else:
            
            section_label("Elite Year Count Distribution")
            if "elite_year_count" in df.columns:
                df_elite = df[df["elite_flag"] == 1]
                fig2 = px.histogram(df_elite, x="elite_year_count", color="segment_label",
                                    color_discrete_map=SEGMENT_COLORS, barmode="overlay",
                                    title=" ",
                                    labels={"elite_year_count": "Years as Elite", "segment_label": ""},
                                    opacity=0.75)
                fig2.update_layout(legend=dict(orientation="h", y=1.1))
                st.plotly_chart(style_fig(fig2, height=350), use_container_width=True)
            else:
                st.info("Kolom elite_year_count tidak tersedia.")

            divider()
            section_label("Elite vs Non-Elite per Segment Comparison")
            if "review_count" in df.columns:
                metric_sel = st.selectbox(
                    "Pilih Metrik",
                    ["user_count", "review_count", "fans"],
                    format_func=lambda x: {
                        "user_count":   "Jumlah User",
                        "review_count": "Total Review",
                        "fans":         "Total Fans",
                    }.get(x, x),
                    key="elite_metric"
                )

                df["tipe"] = df["elite_flag"].map({1: "Elite", 0: "Non-Elite"})

                if metric_sel == "user_count":
                    elite_compare = df.groupby(["segment_label", "tipe"]).size().reset_index(name="nilai")
                else:
                    elite_compare = df.groupby(["segment_label", "tipe"])[metric_sel].sum().reset_index()
                    elite_compare = elite_compare.rename(columns={metric_sel: "nilai"})

                fig3 = px.bar(
                    elite_compare, x="segment_label", y="nilai",
                    title=" ",
                    color="tipe", barmode="group",
                    color_discrete_map={"Elite": "#facc15", "Non-Elite": "#475569"},
                    labels={"segment_label": "", "nilai": "", "tipe": ""},
                )
                fig3.update_layout(legend=dict(orientation="h", y=1.1))
                st.plotly_chart(style_fig(fig3, height=350), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 5 — USER LOOKUP & PREDICT
    # ═══════════════════════════════════════════════════════════════════
    with tab5:
        lookup_tab, predict_tab = st.tabs(["User Lookup", "Predict New User"])

        # ── User Lookup ───────────────────────────────────────────────
        with lookup_tab:
            section_label("Find Existing User")
            search_name = st.text_input("Cari nama user", placeholder="Ketik nama...")
            if not search_name:
                st.info("Masukkan beberapa huruf nama user untuk melihat profilnya.")
            else:
                results = df[df["name"].str.contains(search_name, case=False, na=False)] \
                    if "name" in df.columns else pd.DataFrame()
                if results.empty:
                    st.warning("User tidak ditemukan.")
                else:
                    selected_user = st.selectbox("Pilih User", results["name"].tolist())
                    user_row  = results[results["name"] == selected_user].iloc[0]
                    seg       = user_row.get("segment_label", "-")
                    seg_color = SEGMENT_COLORS.get(seg, "#818cf8")
                    is_elite  = int(float(user_row.get("elite_flag", 0) or 0)) == 1

                    # ── Profile Card ──────────────────────────────────────────
                    st.markdown(f"""
                    <div style="padding:20px 24px;border-radius:12px;background:#1e2130;
                                border:2px solid {seg_color};margin-bottom:16px;
                                display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-size:20px;font-weight:700;color:#e5e7eb;margin-bottom:6px;">
                                {user_row.get('name', '-')}
                            </div>
                            <div style="display:inline-block;padding:4px 12px;border-radius:20px;
                                        background:{seg_color}22;border:1px solid {seg_color};
                                        color:{seg_color};font-size:12px;font-weight:600;">{seg}</div>
                            {"&nbsp;<div style='display:inline-block;padding:4px 12px;border-radius:20px;background:#fbbf2422;border:1px solid #fbbf24;color:#fbbf24;font-size:12px;font-weight:600;'>⭐ Elite</div>" if is_elite else ""}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    divider()
                    section_label(f"Posisi dalam Segment '{seg}'")
                    seg_df = df[df["segment_label"] == seg]

                    display_features = {
                        "review_count":      ("Review",       "{:,.0f}", "review"),
                        "average_stars":     ("Avg Bintang",  "{:.2f}",  "bintang"),
                        "fans":              ("Fans",          "{:,.0f}", "pengikut"),
                        "network_size":      ("Koneksi",       "{:,.0f}", "teman"),
                        "votes_per_review":  ("Votes/Review",  "{:.1f}",  "votes"),
                        "influence_score":   ("Influence",     "{:.1f}",  "skor"),
                        "engagement_score":  ("Engagement",    "{:.1f}",  "skor"),
                        "account_age_years": ("Bergabung",     "{:.0f}",  "tahun lalu"),
                    }

                    # Render 2 kolom
                    feat_items = [
                        (feat, meta) for feat, meta in display_features.items()
                        if feat in df.columns and pd.notna(user_row.get(feat, np.nan))
                    ]

                    for i in range(0, len(feat_items), 2):
                        col_a, col_b = st.columns(2)
                        for col, (feat, (label, fmt, unit)) in zip(
                            [col_a, col_b], feat_items[i:i+2]
                        ):
                            user_val    = float(user_row.get(feat))
                            pct         = (seg_df[feat] <= user_val).mean() * 100
                            val_display = fmt.format(user_val)
                            bar_color   = "#34d399" if pct >= 75 else "#60a5fa" if pct >= 40 else "#9ca3af"

                            with col:
                                st.markdown(f"""
                                <div style="padding:12px 16px;border-radius:10px;background:#1e2130;
                                            border:1px solid #2e3250;margin-bottom:8px;">
                                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                                        <span style="font-size:13px;color:#e5e7eb;font-weight:600;">{label}</span>
                                        <span style="font-size:12px;color:#9ca3af;">{val_display} {unit}
                                            &nbsp;·&nbsp;<span style="color:{bar_color};font-weight:700;">
                                            Top {100 - pct:.0f}%</span>
                                        </span>
                                    </div>
                                    <div style="background:#2e3250;border-radius:999px;height:5px;">
                                        <div style="background:{bar_color};width:{pct:.0f}%;height:5px;
                                                    border-radius:999px;"></div>
                                    </div>
                                </div>""", unsafe_allow_html=True)

        # ── Predict New User ──────────────────────────────────────────
        with predict_tab:
            section_label("Predict Segment — New User")

            if kmeans_model is None:
                st.error("KMeans model tidak berhasil diload.")
            else:
                st.markdown("""<div style="font-size:13px;color:#9ca3af;margin-bottom:16px;">
                    Masukkan data user baru untuk diprediksi segmennya menggunakan model KMeans.
                </div>""", unsafe_allow_html=True)

                input_defaults = {
                    "review_count":          (0,    2000,  50,   1),
                    "reviews_per_year":      (0.0,  50.0,  3.0,  0.1),
                    "average_stars":         (1.0,  5.0,   3.6,  0.1),
                    "rating_deviation":      (0.0,  3.0,   0.8,  0.1),
                    "account_age_days":      (0,    7000,  2000, 1),
                    "votes_per_review":      (0.0,  15.0,  1.0,  0.1),
                    "compliment_per_review": (0.0,  2.0,   0.1,  0.01),
                    "fans":                  (0,    200,   2,    1),
                    "network_size":          (0,    500,   50,   1),
                    "elite_flag":            None,
                    "elite_consistency":     (0.0,  1.0,   0.0,  0.01),
                    "engagement_score":      (0.0,  10.0,  1.0,  0.1),
                    "influence_score":       (0.0,  0.2,   0.01, 0.001),
                }

                st.markdown("<div style='font-size:12px;color:#9ca3af;margin-bottom:8px;'>Autofill contoh profil:</div>", unsafe_allow_html=True)
                af1, af2, af3, af4 = st.columns(4)

                AUTOFILL_PRESETS = {
                    "Highly Eng Influencers": {
                        "review_count": 200, "reviews_per_year": 12.0, "average_stars": 3.8,
                        "rating_deviation": 0.5, "account_age_days": 6000, "votes_per_review": 3.3,
                        "compliment_per_review": 0.4, "fans": 20, "network_size": 150,
                        "elite_flag": 1, "elite_consistency": 0.12, "engagement_score": 4.0,
                        "influence_score": 0.05,
                    },
                    "Active Contributors": {
                        "review_count": 75, "reviews_per_year": 5.5, "average_stars": 3.7,
                        "rating_deviation": 0.7, "account_age_days": 4900, "votes_per_review": 2.0,
                        "compliment_per_review": 0.2, "fans": 5, "network_size": 70,
                        "elite_flag": 0, "elite_consistency": 0.06, "engagement_score": 2.1,
                        "influence_score": 0.02,
                    },
                    "Regular Reviewers": {
                        "review_count": 40, "reviews_per_year": 3.8, "average_stars": 3.6,
                        "rating_deviation": 0.85, "account_age_days": 3900, "votes_per_review": 1.3,
                        "compliment_per_review": 0.1, "fans": 3, "network_size": 70,
                        "elite_flag": 0, "elite_consistency": 0.05, "engagement_score": 1.4,
                        "influence_score": 0.01,
                    },
                    "Casual Users": {
                        "review_count": 20, "reviews_per_year": 2.7, "average_stars": 3.4,
                        "rating_deviation": 1.1, "account_age_days": 2650, "votes_per_review": 0.9,
                        "compliment_per_review": 0.08, "fans": 1, "network_size": 40,
                        "elite_flag": 0, "elite_consistency": 0.03, "engagement_score": 0.8,
                        "influence_score": 0.01,
                    },
                }

                for col, seg in zip([af1, af2, af3, af4], AUTOFILL_PRESETS.keys()):
                    with col:
                        if st.button(seg, key=f"autofill_{seg}", use_container_width=True):
                            for feat, val in AUTOFILL_PRESETS[seg].items():
                                st.session_state[f"inp_{feat}"] = val
                            st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)

                inputs = {}
                cols3  = st.columns(3)
                for i, feat in enumerate(avail_features):
                    with cols3[i % 3]:
                        label = FEATURE_LABELS.get(feat, feat)
                        if feat == "elite_flag":
                            inputs[feat] = st.selectbox(
                                label, [0, 1],
                                index=int(st.session_state.get(f"inp_{feat}", 0)),
                                key=f"inp_{feat}",
                                help=FEATURE_DESCRIPTIONS.get(feat, "")
                            )
                        else:
                            mn, mx, dv, step = input_defaults.get(feat, (0.0, 100.0, 1.0, 0.1))
                            inputs[feat] = st.number_input(
                                label,
                                min_value=float(mn),
                                max_value=float(mx),
                                value=float(st.session_state.get(f"inp_{feat}", dv)),
                                step=float(step),
                                key=f"inp_{feat}",
                                help=FEATURE_DESCRIPTIONS.get(feat, "")
                            )

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Predict Segment", use_container_width=True):
                    try:
                        input_array   = pd.DataFrame([[inputs[f] for f in avail_features]], columns=avail_features)
                        cluster_id    = int(kmeans_model.predict(input_array)[0])
                        predicted_seg = CLUSTER_MAPPING.get(cluster_id, f"Cluster {cluster_id}")
                        seg_color     = SEGMENT_COLORS.get(predicted_seg, "#818cf8")

                        st.markdown(f"""
                        <div style="padding:24px;border-radius:12px;background:#1e2130;
                                    border:2px solid {seg_color};text-align:center;margin-top:16px;">
                            <div style="font-size:13px;color:#9ca3af;margin-bottom:8px;">PREDICTED SEGMENT</div>
                            <div style="font-size:28px;font-weight:700;color:{seg_color};">{predicted_seg}</div>
                            <div style="font-size:12px;color:#6b7280;margin-top:6px;">KMeans Cluster ID: {cluster_id}</div>
                        </div>""", unsafe_allow_html=True)

                        divider()
                        section_label(f"Input User vs Avg '{predicted_seg}'")
                        seg_avg   = df[df["segment_label"] == predicted_seg][avail_features].mean()

                        EXCLUDE_FROM_CHART = {"account_age_days"}

                        comp_rows = [{
                            "Feature":     FEATURE_LABELS.get(f, f),
                            "Input User":  round(inputs[f], 2),
                            "Segment Avg": round(float(seg_avg[f]), 2),
                        } for f in avail_features if f not in EXCLUDE_FROM_CHART]
                        comp_df = pd.DataFrame(comp_rows)

                        # Tampilkan account_age_days sebagai KPI card
                        if "account_age_days" in avail_features:
                            age_val     = round(inputs["account_age_days"] / 365, 1)
                            age_avg     = round(float(seg_avg["account_age_days"]) / 365, 1)
                            st.markdown(f"""
                            <div style="padding:12px 16px;border-radius:10px;background:#1e2130;
                                        border:1px solid #2e3250;margin-bottom:12px;
                                        display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-size:13px;color:#e5e7eb;font-weight:600;">Account Age</span>
                                <span style="font-size:13px;color:#9ca3af;">
                                    Input: <b style="color:#e5e7eb;">{age_val} tahun</b>
                                    &nbsp;·&nbsp;
                                    Avg {predicted_seg}: <b style="color:#9ca3af;">{age_avg} tahun</b>
                                </span>
                            </div>""", unsafe_allow_html=True)

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name="Input User", x=comp_df["Feature"], y=comp_df["Input User"],
                            marker_color=seg_color, opacity=0.9,
                        ))
                        fig.add_trace(go.Bar(
                            name=f"Avg {predicted_seg}", x=comp_df["Feature"], y=comp_df["Segment Avg"],
                            marker_color="#475569", opacity=0.7,
                        ))
                        fig.update_layout(
                            barmode="group",
                            xaxis=dict(tickangle=-30, gridcolor=GRID_COLOR),
                            yaxis=dict(gridcolor=GRID_COLOR),
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(style_fig(fig, height=400), use_container_width=True)

                    except Exception as e:
                        st.error(f"Prediksi gagal: {e}")
