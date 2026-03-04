import streamlit as st
import html as html_lib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.style import apply_style, page_header, section_label, divider, style_fig, FONT_COLOR, GRID_COLOR

STAR_COLOR  = "#fbbf24"
SCORE_COLOR = "#818cf8"


def kpi_card(label, value, sub="", color="#60a5fa"):
    fs = "18px" if len(str(value)) > 10 else "24px"
    sub_html = f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="padding:16px;border-radius:10px;background:#1e2130;border:1px solid #2e3250;
                text-align:center;min-height:110px;display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
        <div style="font-size:{fs};font-weight:700;color:{color};">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def star_display(stars):
    filled = int(round(float(stars))) if pd.notna(stars) else 0
    return "★" * filled + "☆" * (5 - filled)


def render_biz_card(row, rank=None, show_score=True):
    import html as html_lib
    name       = html_lib.escape(str(row.get("name") or row.get("business_id", "Unknown"))[:50])
    # Pakai mapped_category kalau ada, fallback ke categories
    cat_raw    = row.get("mapped_category") or row.get("categories", "-")
    categories = html_lib.escape(str(cat_raw)[:40])
    city       = html_lib.escape(str(row.get("city", "-")))
    stars      = float(row.get("stars", 0) or 0)
    score      = row.get("predicted_score", None)
    filled     = int(round(stars))
    star_str   = "★" * filled + "☆" * (5 - filled)

    rank_badge = f'<span style="font-size:10px;font-weight:700;color:#4b5563;background:#0e1117;padding:2px 6px;border-radius:6px;border:1px solid #2e3250;">#{rank}</span>' if rank else ""
    def get_match_badge(score):
        if score >= 0.7:
            return ("Perfect Match", "#34d399", "#0d2b1f")
        elif score >= 0.5:
            return ("Great Match", "#818cf8", "#1a1b35")
        elif score >= 0.3:
            return ("Good Match", "#fbbf24", "#2b2010")
        else:
            return ("Possible Match", "#6b7280", "#1a1b1f")

    if show_score and score is not None:
        label, color, bg = get_match_badge(score)
        score_row = f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #2e3250;display:flex;justify-content:flex-end;"><span style="font-size:10px;font-weight:700;color:{color};background:{bg};padding:3px 10px;border-radius:999px;border:1px solid {color}44;">✦ {label}</span></div>'
    else:
        score_row = ""

    st.markdown(f"""<div style="padding:16px;border-radius:14px;background:linear-gradient(135deg,#1e2130,#161929);border:1px solid #2e3250;margin-bottom:10px;box-shadow:0 4px 20px rgba(0,0,0,0.3);min-height:160px;display:flex;flex-direction:column;"><div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">{rank_badge}<div style="display:inline-block;padding:2px 10px;border-radius:999px;background:#818cf822;border:1px solid #818cf844;font-size:10px;color:#818cf8;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{categories}</div></div><div style="font-size:14px;font-weight:700;color:#e5e7eb;line-height:1.4;margin-bottom:8px;min-height:42px;">{name}</div><div style="font-size:14px;color:#fbbf24;margin-bottom:6px;">{star_str} <span style="font-size:12px;color:#9ca3af;">{stars:.1f}</span></div><div style="font-size:11px;color:#6b7280;">📍 {city}</div>{score_row}</div>""", unsafe_allow_html=True)


def render(ncf_model=None, le_user=None, le_biz=None, biz_features=None, biz_info=None, df_slim=None, df_user=None, df_business=None):
    apply_style()
    page_header("Recommendation System", "Rekomendasi bisnis data Yelp yang dipersonalisasi menggunakan Hybrid Neural Collaborative Filtering")
    st.warning("Model di Sistem Rekomendasi masih dalam tahap pengembangan dan belum dioptimalkan sepenuhnya. Hasil yang ditampilkan hanya untuk keperluan eksplorasi. Masih terdapat beberapa kesalahan dalam memberikan rekomendasi bisnis")

    try:
        from models.recsys.loader import get_recommendations, get_user_history
    except ImportError:
        st.error("models/recsys/loader.py tidak ditemukan.")
        return

    if df_business is not None and biz_info is not None:
        enrich   = df_business[["business_id", "name", "categories", "city"]].drop_duplicates("business_id")
        biz_info = biz_info.merge(enrich, on="business_id", how="left", suffixes=("", "_biz"))
        for col in ["name", "categories", "city"]:
            col_biz = col + "_biz"
            if col_biz in biz_info.columns:
                if col in biz_info.columns:
                    biz_info[col] = biz_info[col].fillna(biz_info[col_biz])
                else:
                    biz_info[col] = biz_info[col_biz]
                biz_info = biz_info.drop(columns=[col_biz])

    if ncf_model is None:
        st.error("Hybrid NCF model tidak berhasil diload. Pastikan hybrid_ncf_model.pt tersedia di models/recsys/.")
        return

    # ── Name ↔ User ID lookup ─────────────────────────────────────────────────
    name_to_uid = {}
    uid_to_name = {}
    if df_user is not None and "name" in df_user.columns and "user_id" in df_user.columns:
        known       = set(le_user.classes_) if le_user is not None else set()
        df_known    = df_user[df_user["user_id"].isin(known)]
        name_to_uid = dict(zip(df_known["name"], df_known["user_id"]))
        uid_to_name = dict(zip(df_known["user_id"], df_known["name"]))

    # ── KPI ───────────────────────────────────────────────────────────────────
    n_users = len(le_user.classes_) if le_user else 0
    n_items = len(le_biz.classes_)  if le_biz  else 0
    n_inter = len(df_slim)           if df_slim is not None else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total Users",        f"{n_users:,}",  "in model",              "#60a5fa")
    with k2: kpi_card("Total Businesses",   f"{n_items:,}",  "recommendable",         "#818cf8")
    with k3: kpi_card("Total Interactions", f"{n_inter:,}",  "user-biz pairs",        "#34d399")
    with k4: kpi_card("Model",              "Hybrid NCF",    "GMF + MLP + Content",   "#fbbf24")

    divider()

    tab1, tab2 = st.tabs(["Get Recommendations", "Explore Businesses"])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — GET RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        section_label("Find User & Generate Recommendations")

        c_input, c_opt = st.columns([2, 2])
        with c_input:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            search_name = st.text_input(
                "Search User by Name",
                placeholder="Type a name to search...",
                label_visibility="collapsed",
                key="recsys_name_search",
            )
        with c_opt:
            top_n = st.slider("Top N", min_value=5, max_value=20, value=10, step=5)

        # Name search → selectbox
        user_id_input = ""
        if search_name:
            matched_names = [n for n in name_to_uid if search_name.lower() in n.lower()]
            if matched_names:
                chosen_name   = st.selectbox("Select User", matched_names, key="recsys_name_select")
                user_id_input = name_to_uid[chosen_name]
                st.caption(f"User ID: `{user_id_input}`")
            else:
                st.warning("No user found with that name.")

        # Quick pick — most active users
        if df_slim is not None:
            top_users = df_slim.groupby("user_id").size().sort_values(ascending=False).head(5)
            st.markdown(
                "<div style='font-size:12px;color:#6b7280;margin:8px 0 6px;'>Quick pick — most active users:</div>",
                unsafe_allow_html=True,
            )
            qcols = st.columns(5)
            for col, (uid, cnt) in zip(qcols, top_users.items()):
                with col:
                    display_name = uid_to_name.get(uid, uid[:12])
                    if st.button(
                        f"{display_name[:14]}\n({cnt} reviews)",
                        key=f"quick_{uid}",
                        use_container_width=True,
                    ):
                        st.session_state["recsys_user_id"] = uid
                        st.rerun()

        # Use session state if quick-picked
        if "recsys_user_id" in st.session_state and not user_id_input:
            user_id_input = st.session_state["recsys_user_id"]

        if not user_id_input:
            st.info("Search for a user by name above to generate personalized recommendations.")
        else:
            uid          = user_id_input.strip()
            display_name = uid_to_name.get(uid, uid)

            if le_user is not None and uid not in le_user.classes_:
                st.warning(f"User `{display_name}` not found in the model.")
            else:
                history_df = get_user_history(df_slim, le_biz, biz_info, uid)

                # ── Enrich history_df dengan mapped_category ──────────────────────────────
                if "mapped_category" not in history_df.columns or history_df["mapped_category"].isna().all():
                    mc_source = None
                    if biz_features is not None and "mapped_category" in biz_features.columns:
                        mc_source = biz_features[["business_id", "mapped_category"]].drop_duplicates("business_id")
                    elif biz_info is not None and "mapped_category" in biz_info.columns:
                        mc_source = biz_info[["business_id", "mapped_category"]].drop_duplicates("business_id")

                    if mc_source is not None:
                        history_df = history_df.drop(columns=["mapped_category"], errors="ignore")
                        history_df = history_df.merge(mc_source, on="business_id", how="left")

                n_hist = len(history_df)

                u1, u2, u3 = st.columns(3)
                with u1: kpi_card("User",         display_name[:20], uid[:20],              "#60a5fa")
                with u2: kpi_card("Past Reviews", str(n_hist),       "in interaction data", "#34d399")
                with u3:
                    avg_stars = history_df["stars"].mean() if n_hist > 0 else 0
                    kpi_card("Avg Rating Given", f"{avg_stars:.1f} ★", "across reviews", STAR_COLOR)

                divider()

                # ── Past Reviews ──────────────────────────────────────────────
                if n_hist > 0:
                    section_label(f"{display_name}'s Past Reviews")
                    show_n    = min(n_hist, 8)
                    history_top = history_df.head(show_n)
                    for i in range(0, show_n, 4):
                        row_cols = st.columns(4)
                        for col, (_, r) in zip(row_cols, history_top.iloc[i:i+4].iterrows()):
                            with col:
                                render_biz_card(r, show_score=False)
                    divider()

                # ── Recommendations ───────────────────────────────────────────
                section_label(f"Recommendations for {display_name}")

                with st.spinner("Generating recommendations..."):
                    rec_df = get_recommendations(
                        ncf_model, le_user, le_biz, biz_features,
                        biz_info, df_slim, uid, top_n=top_n, exclude_seen=True,
                    )

                if rec_df.empty:
                    st.warning("No recommendations generated.")
                else:
                    # ── Normalisasi skor 85–99% ───────────────────────────────
                    max_raw = rec_df["predicted_score"].max()
                    min_raw = rec_df["predicted_score"].min()

                    def normalize_score(s):
                        if max_raw == min_raw: return 95.0
                        return 85 + ((s - min_raw) / (max_raw - min_raw)) * 14

                    rec_df["display_score"] = rec_df["predicted_score"].apply(normalize_score)

                    # ── Top Category: pakai mapped_category kalau ada ─────────
                    if "mapped_category" in rec_df.columns and rec_df["mapped_category"].notna().any():
                        top_cat   = rec_df["mapped_category"].value_counts().idxmax()
                        cat_stats = rec_df["mapped_category"].value_counts().head(5)
                    else:
                        all_rec_cats = rec_df["categories"].str.split(", ").explode()
                        top_cat      = all_rec_cats.value_counts().idxmax()
                        cat_stats    = all_rec_cats.value_counts().head(5)

                    # ── Enrich dari df_business yang sudah pasti punya name & city ──
                    if df_business is not None:
                        enrich_cols = [c for c in ["business_id", "name", "city", "stars"] if c in df_business.columns]
                        rec_df = rec_df.drop(columns=[c for c in ["name", "city", "stars"] if c in rec_df.columns], errors="ignore")
                        rec_df = rec_df.merge(
                            df_business[enrich_cols].drop_duplicates("business_id"),
                            on="business_id", how="left"
                        )

                    # ── Personalized Analysis Card ────────────────────────────
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e2130 0%, #111420 100%); padding: 25px; border-radius: 20px; border: 1px solid #343950; margin-bottom: 30px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h3 style="margin:0; color:#818cf8; font-size:20px;">Personalized Analysis</h3>
                                <p style="color:#9ca3af; font-size:14px; margin-top:5px;">We analyzed {len(le_biz.classes_)} places to find your perfect match.</p>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 32px; font-weight: 800; color: #34d399;">{rec_df['display_score'].max():.1f}%</span>
                                <div style="font-size: 10px; color: #6b7280; text-transform: uppercase;">Highest Match Score</div>
                            </div>
                        </div>
                        <div style="margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                            <div style="background: #0e1117; padding: 12px; border-radius: 12px; border: 1px solid #2e3250; text-align: center;">
                                <div style="color: #6b7280; font-size: 10px;">TOP CATEGORY</div>
                                <div style="color: #fbbf24; font-size: 14px; font-weight: 600;">{top_cat}</div>
                            </div>
                            <div style="background: #0e1117; padding: 12px; border-radius: 12px; border: 1px solid #2e3250; text-align: center;">
                                <div style="color: #6b7280; font-size: 10px;">DIVERSITY</div>
                                <div style="color: #818cf8; font-size: 14px; font-weight: 600;">{rec_df['city'].nunique() if 'city' in rec_df.columns else '-'} Cities</div>
                            </div>
                            <div style="background: #0e1117; padding: 12px; border-radius: 12px; border: 1px solid #2e3250; text-align: center;">
                                <div style="color: #6b7280; font-size: 10px;">RECOMMENDATIONS</div>
                                <div style="color: #34d399; font-size: 14px; font-weight: 600;">{len(rec_df)} Places</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Radar Chart (Vibe) ────────────────────────────────────
                    fig_vibe = go.Figure(data=go.Scatterpolar(
                        r=cat_stats.values,
                        theta=cat_stats.index,
                        fill='toself',
                        line=dict(color='#818cf8')
                    ))
                    fig_vibe.update_layout(
                        polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
                        title=" ",
                        showlegend=False, height=250, margin=dict(t=30, b=30, l=30, r=30)
                    )

                    col_chart, col_info = st.columns([1.5, 1])
                    with col_chart:
                        st.plotly_chart(style_fig(fig_vibe), use_container_width=True)
                    with col_info:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.write("**Your Recommendation Vibe**")
                        st.caption("The shape of this graph shows the 'personality' of our suggestions for you.")

                    divider()
                    section_label("Recommendation Details")

                    for i in range(0, len(rec_df), 3):
                        cols = st.columns(3)
                        for col, (_, row) in zip(cols, rec_df.iloc[i:i+3].iterrows()):
                            with col:
                                render_biz_card(row, rank=int(row["rank"]), show_score=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — EXPLORE BUSINESSES
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        if biz_info is None:
            st.info("biz_info tidak tersedia.")
        else:
            section_label("Business Overview")

            k1, k2, k3 = st.columns(3)
            with k1: kpi_card("Total Businesses", f"{len(biz_info):,}", "in catalog",            "#60a5fa")
            with k2:
                avg_s = biz_info["stars"].mean()
                kpi_card("Avg Rating", f"{avg_s:.2f} ★", "across all businesses", STAR_COLOR)
            with k3:
                n_cities = biz_info["city"].nunique() if "city" in biz_info.columns else 0
                kpi_card("Cities", str(n_cities), "unique cities", "#34d399")

            divider()
            c1, c2 = st.columns(2)

            with c1:
                section_label("Rating Distribution")
                star_counts = biz_info["stars"].value_counts().sort_index().reset_index()
                star_counts.columns = ["stars", "count"]
                fig_stars = go.Figure(go.Bar(
                    x=star_counts["stars"],
                    y=star_counts["count"],
                    marker_color=STAR_COLOR, opacity=0.85,
                    text=star_counts["count"], textposition="outside",
                    hovertemplate="<b>%{x} Stars</b><br>%{y:,} businesses<extra></extra>",
                ))
                fig_stars.update_layout(
                    title=" ",
                    xaxis=dict(title="Stars", gridcolor=GRID_COLOR),
                    yaxis=dict(title="Businesses", gridcolor=GRID_COLOR),
                    showlegend=False,
                )
                st.plotly_chart(style_fig(fig_stars, height=300), use_container_width=True)

            with c2:
                section_label("Top Cities")
                city_counts = biz_info["city"].value_counts().head(10).reset_index()
                city_counts.columns = ["city", "count"]
                fig_city = go.Figure(go.Bar(
                    x=city_counts["count"],
                    y=city_counts["city"],
                    orientation="h",
                    marker_color="#60a5fa", opacity=0.85,
                    text=city_counts["count"], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>%{x:,} businesses<extra></extra>",
                ))
                fig_city.update_layout(
                    title=" ",
                    xaxis=dict(title="Businesses", gridcolor=GRID_COLOR),
                    yaxis=dict(title="", autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(style_fig(fig_city, height=300), use_container_width=True)

            divider()
            section_label("Search Business Catalog")

            # Ambil mapped_category dari biz_features (BizLookup) yang sudah ada mapped_category-nya
            if biz_features is not None and "mapped_category" in biz_features.columns:
                cat_options = sorted(biz_features["mapped_category"].dropna().unique().tolist())
            elif "mapped_category" in biz_info.columns:
                cat_options = sorted(biz_info["mapped_category"].dropna().unique().tolist())
            else:
                cat_options = sorted(
                    biz_info["categories"].dropna()
                    .str.split(",").explode()
                    .str.strip().unique().tolist()
                )

            # Pastikan biz_info punya mapped_category dari biz_features
            if biz_features is not None and "mapped_category" in biz_features.columns:
                if "mapped_category" not in biz_info.columns:
                    mc = biz_features[["business_id","mapped_category"]].drop_duplicates("business_id")
                    biz_info = biz_info.merge(mc, on="business_id", how="left")

            filter_df = biz_info.copy()

            # ── Baris 1: hanya category ──
            search_cat = st.selectbox("Category", [""] + cat_options,
                                    format_func=lambda x: "Select a category..." if x == "" else x,
                                    key="biz_cat")

            if search_cat:
                if "mapped_category" in filter_df.columns:
                    filter_df = filter_df[filter_df["mapped_category"] == search_cat]
                else:
                    filter_df = filter_df[filter_df["categories"].str.contains(search_cat, case=False, na=False)]

                city_options = sorted(filter_df["city"].dropna().unique().tolist()) if "city" in filter_df.columns else []

                # ── Baris 2: category + city ──
                c1, c2 = st.columns(2)
                with c1:
                    st.selectbox("Category", [""] + cat_options,
                                format_func=lambda x: "Select a category..." if x == "" else x,
                                key="biz_cat_2", index=cat_options.index(search_cat) + 1,
                                disabled=True, label_visibility="collapsed")
                with c2:
                    search_city = st.selectbox("City", [""] + city_options,
                                                format_func=lambda x: "All cities..." if x == "" else x,
                                                key="biz_city", label_visibility="collapsed")

                if search_city:
                    filter_df = filter_df[filter_df["city"] == search_city]

                    # ── Baris 3: category + city + name ──
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.selectbox("Category", [""] + cat_options,
                                    format_func=lambda x: x,
                                    key="biz_cat_3", index=cat_options.index(search_cat) + 1,
                                    disabled=True, label_visibility="collapsed")
                    with c2:
                        st.selectbox("City", [""] + city_options,
                                    format_func=lambda x: x,
                                    key="biz_city_3", index=city_options.index(search_city) + 1,
                                    disabled=True, label_visibility="collapsed")
                    with c3:
                        search_biz_name = st.text_input("Name", placeholder="Narrow by name...",
                                                        label_visibility="collapsed", key="biz_name")
                    if search_biz_name:
                        filter_df = filter_df[filter_df["name"].str.contains(search_biz_name, case=False, na=False)]
                else:
                    search_biz_name = ""

            else:
                search_city     = ""
                search_biz_name = ""

            st.caption(f"Showing {min(len(filter_df), 12):,} of {len(filter_df):,} businesses")

            # ── Grid card ──
            show_df = filter_df.head(12)
            for i in range(0, len(show_df), 3):
                cols = st.columns(3)
                for col, (_, row) in zip(cols, show_df.iloc[i:i+3].iterrows()):
                    with col:
                        name     = str(row.get("name") or row.get("business_id", "-"))[:40]
                        city     = str(row.get("city", "-"))
                        cats     = str(row.get("mapped_category") or row.get("categories", "-"))[:50]
                        stars    = float(row.get("stars", 0) or 0)
                        filled   = int(round(stars))
                        star_str = "★" * filled + "☆" * (5 - filled)
                        st.markdown(f"""
                        <div style="padding:14px;border-radius:12px;background:#1e2130;
                                    border:1px solid #2e3250;margin-bottom:10px;
                                    box-shadow:0 2px 12px rgba(0,0,0,0.3);">
                            <div style="font-size:10px;color:#818cf8;margin-bottom:6px;
                                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                                {cats}
                            </div>
                            <div style="font-size:13px;font-weight:700;color:#e5e7eb;
                                        margin-bottom:6px;min-height:36px;">
                                {name}
                            </div>
                            <div style="font-size:13px;color:#fbbf24;">
                                {star_str} <span style="font-size:11px;color:#9ca3af;">{stars:.1f}</span>
                            </div>
                            <div style="font-size:11px;color:#6b7280;margin-top:4px;">📍 {city}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if search_cat and len(filter_df) > 12:
                st.caption("Refine your search to see more specific results.")