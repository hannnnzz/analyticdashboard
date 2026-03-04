import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.style import apply_style, page_header, section_label, divider, style_fig, FONT_COLOR, GRID_COLOR

CHURN_COLORS = {
    "Churn":    "#f87171",
    "Retained": "#34d399",
}

FEATURE_LABELS = {
    "total_reviews":           "Total Review",
    "avg_stars":               "Avg Rating",
    "total_useful":            "Total Useful",
    "total_funny":             "Total Funny",
    "total_cool":              "Total Cool",
    "avg_sentiment_score":     "Avg Sentiment Score",
    "avg_review_length":       "Review Length",
    "days_since_last_review":  "Day Since Last Review",
    "days_since_first_review": "Day Since First Review",
    "votes_per_review":        "Vote per Review",
}

FEATURE_DESCRIPTIONS = {
    "total_reviews":           "How many times the user has written a review",
    "avg_stars":               "Average star rating given by the user (1–5)",
    "total_useful":            "How many times the user's reviews were voted as useful by others",
    "total_funny":             "How many times the user's reviews were voted as funny",
    "total_cool":              "How many times the user's reviews were voted as cool",
    "avg_sentiment_score":     "How positive the words in the user's reviews are (0 = negative, 1 = positive)",
    "avg_review_length":       "Average number of characters per review",
    "days_since_last_review":  "How many days since the user last wrote a review",
    "days_since_first_review": "How many days since the user first joined",
    "votes_per_review":        "Average number of votes received per single review",
}

FEATURES = list(FEATURE_LABELS.keys())

INPUT_DEFAULTS = {
    "total_reviews":          (1,    500,   10,   1),
    "avg_stars":              (1.0,  5.0,   3.5,  0.1),
    "total_useful":           (0,    500,   5,    1),
    "total_funny":            (0,    200,   1,    1),
    "total_cool":             (0,    200,   1,    1),
    "avg_sentiment_score":    (0.0,  1.0,   0.5,  0.01),
    "avg_review_length":      (10.0, 2000.0,150.0,10.0),
    "days_since_last_review": (0,    5000,  500,  1),
    "days_since_first_review":(0,    6000,  1000, 1),
    "votes_per_review":       (0.0,  20.0,  1.0,  0.1),
}


def kpi_card(label, value, sub="", color="#60a5fa", border=False):
    border_css = "border:1px solid #2e3250;"
    fs = "18px" if len(str(value)) > 10 else "24px"
    sub_html = f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="padding:16px;border-radius:10px;background:#1e2130;{border_css}
                text-align:center;min-height:120px;display:flex;
                flex-direction:column;justify-content:center;">
        <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
        <div style="font-size:{fs};font-weight:700;color:{color};">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def stat_delta_card(label, val_churn, val_retained):
    delta     = val_retained - val_churn
    pct       = (delta / (abs(val_churn) + 1e-9)) * 100
    direction = "lebih tinggi" if delta > 0 else "lebih rendah"
    color     = "#34d399" if delta > 0 else "#f87171"
    arrow     = "&#9650;" if delta > 0 else "&#9660;"
    st.markdown(f"""
    <div style="padding:16px;border-radius:12px;background:#1e2130;
                border:1px solid #2e3250;margin-bottom:10px;">
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:12px;">{label}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="text-align:center;flex:1;">
                <div style="font-size:10px;color:#f87171;margin-bottom:3px;">Churn</div>
                <div style="font-size:20px;font-weight:700;color:#f87171;">{val_churn:.1f}</div>
            </div>
            <div style="text-align:center;flex:1;">
                <div style="font-size:16px;color:{color};font-weight:700;">{arrow} {abs(pct):.0f}%</div>
                <div style="font-size:10px;color:#6b7280;">{direction}</div>
            </div>
            <div style="text-align:center;flex:1;">
                <div style="font-size:10px;color:#34d399;margin-bottom:3px;">Retained</div>
                <div style="font-size:20px;font-weight:700;color:#34d399;">{val_retained:.1f}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def render(df_churn, df_user=None, churn_model=None):
    apply_style()
    page_header("User Churn Overview", "Memahami pengguna yang berhenti aktif di platform Yelp")

    df = df_churn.copy()
    if "churn" not in df.columns:
        st.warning("Kolom 'churn' tidak ditemukan di dataset.")
        return

    df["churn_label"] = df["churn"].map({1: "Churn", 0: "Retained"})
    avail_features    = [f for f in FEATURES if f in df.columns]

    total      = len(df)
    n_churn    = int(df["churn"].sum())
    n_retained = total - n_churn
    churn_rate = n_churn / total * 100

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total User",   f"{total:,}",        "in dataset",     "#60a5fa")
    with k2: kpi_card("Churn",      f"{n_churn:,}",      "user churn",    "#f87171", border=True)
    with k3: kpi_card("Retained",      f"{n_retained:,}",   "user retained", "#34d399", border=True)
    with k4: kpi_card("Churn Rate",    f"{churn_rate:.1f}%","from total",        "#fbbf24")

    divider()
    tab1, tab2, tab3 = st.tabs(["Churn Overview", "Feature Analysis", "Churn Prediction"])


    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 — GAMBARAN UMUM
    # ═══════════════════════════════════════════════════════════════════
    with tab1:

        c1, c2 = st.columns([1, 1.7])

        with c1:
            section_label("User Proportion")
            seg = df["churn_label"].value_counts().reset_index()
            seg.columns = ["label", "count"]
            fig_pie = go.Figure(go.Pie(
                labels=seg["label"], values=seg["count"], hole=0.62,
                marker=dict(
                    colors=[CHURN_COLORS.get(s, "#9ca3af") for s in seg["label"]],
                    line=dict(width=0),
                ),
                textinfo="percent+label",
                textfont=dict(color="#e5e7eb", size=13),
                hovertemplate="<b>%{label}</b><br>%{value:,} pengguna<br>%{percent}<extra></extra>",
                direction="clockwise",
                sort=False,
            ))
            fig_pie.update_layout(
                showlegend=False,
                title=" ",
                annotations=[dict(
                    text=f"<b>{churn_rate:.1f}%</b><br>Churn",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color="#f87171", size=20),
                )],
            )
            st.plotly_chart(style_fig(fig_pie, height=320), use_container_width=True)

        with c2:
            section_label("Average Behavior Comparison")
            key_feats = [f for f in [
                "total_reviews", "avg_stars", "avg_sentiment_score",
                "avg_review_length", "days_since_last_review", "votes_per_review",
            ] if f in avail_features]

            avg_data = df.groupby("churn_label")[key_feats].mean()
            churn_row    = avg_data.loc["Churn"]    if "Churn"    in avg_data.index else None
            retained_row = avg_data.loc["Retained"] if "Retained" in avg_data.index else None

            if churn_row is not None and retained_row is not None:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="Churn",
                    y=[FEATURE_LABELS.get(f, f) for f in key_feats],
                    x=[float(churn_row[f]) for f in key_feats],
                    orientation="h",
                    marker_color="#f87171", opacity=0.85,
                    text=[f"{float(churn_row[f]):.1f}" for f in key_feats],
                    textposition="inside", insidetextanchor="end",
                    hovertemplate="<b>%{y}</b><br>Churn avg: %{x:.2f}<extra></extra>",
                ))
                fig_bar.add_trace(go.Bar(
                    name="Retained",
                    y=[FEATURE_LABELS.get(f, f) for f in key_feats],
                    x=[float(retained_row[f]) for f in key_feats],
                    orientation="h",
                    marker_color="#34d399", opacity=0.85,
                    text=[f"{float(retained_row[f]):.1f}" for f in key_feats],
                    textposition="inside", insidetextanchor="end",
                    hovertemplate="<b>%{y}</b><br>Retained avg: %{x:.2f}<extra></extra>",
                ))
                fig_bar.update_layout(
                    barmode="group",
                    title=" ",
                    xaxis=dict(title="Nilai Rata-rata", gridcolor=GRID_COLOR),
                    yaxis=dict(title="", autorange="reversed"),
                    legend=dict(orientation="h", y=1.08, x=0),
                    margin=dict(l=0, r=10, t=40, b=20),
                )
                st.plotly_chart(style_fig(fig_bar, height=320), use_container_width=True)


    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 — ANALISIS FITUR
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        section_label("What Differentiates Churned vs Retained Users?")

        col_sel, col_desc = st.columns([1, 2])
        feat_options = {FEATURE_LABELS.get(f, f): f for f in avail_features}

        with col_sel:
            st.markdown("<div style='margin-top:4px;'>", unsafe_allow_html=True)
            chosen_label = st.selectbox("Select Feature", list(feat_options.keys()), key="tab2_feat", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

        feat_dist = feat_options[chosen_label]

        with col_desc:
            st.markdown(
                f"<div style='padding:10px 14px;background:#1e2130;border-radius:8px;"
                f"border-left:3px solid #60a5fa;font-size:13px;color:#9ca3af;'>"
                f"<span style='font-size:11px;color:#6b7280;text-transform:uppercase;"
                f"letter-spacing:0.08em;'>Feature Description</span><br><br>"
                f"{FEATURE_DESCRIPTIONS.get(feat_dist, '')}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        churn_vals    = df[df["churn"] == 1][feat_dist].dropna()
        retained_vals = df[df["churn"] == 0][feat_dist].dropna()

        avg_c = churn_vals.mean()
        avg_r = retained_vals.mean()
        diff  = avg_r - avg_c
        pct   = (diff / (abs(avg_c) + 1e-9)) * 100
        winner = "Retained" if avg_r > avg_c else "Churn"
        w_color = "#34d399" if winner == "Retained" else "#f87171"
        arrow  = "▲" if diff > 0 else "▼"

        # Summary insight banner
        st.markdown(f"""
        <div style="padding:16px 24px;border-radius:12px;background:#1e2130;
                    border:1px solid #2e3250;margin-bottom:20px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:4px;">Key Insight</div>
                <div style="font-size:15px;color:#e5e7eb;font-weight:600;">
                    Retained users have
                    <span style="color:{w_color};">{arrow} {abs(pct):.0f}% {"higher" if diff > 0 else "lower"}</span>
                    {chosen_label.lower()} compared to churned users.
                </div>
            </div>
            <div style="font-size:28px;">{("📈" if diff > 0 else "📉")}</div>
        </div>
        """, unsafe_allow_html=True)

        # Big comparison cards
        k1, k2, k3 = st.columns(3)
        with k1:
            kpi_card("Avg — Churn",    f"{avg_c:.2f}", "churned users",    "#f87171")
        with k2:
            kpi_card("Avg — Retained", f"{avg_r:.2f}", "retained users",   "#34d399")
        with k3:
            diff_color = "#34d399" if diff > 0 else "#f87171"
            kpi_card("Difference", f"{arrow} {abs(diff):.2f}", f"{abs(pct):.0f}% gap", diff_color)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-side avg bar — clean and simple
        c1, c2 = st.columns([1.4, 1])

        with c1:
            section_label("Average Value Comparison")
            fig_avg = go.Figure()
            for group, val, color in [("Churn", avg_c, "#f87171"), ("Retained", avg_r, "#34d399")]:
                fig_avg.add_trace(go.Bar(
                    name=group, x=[group], y=[val],
                    marker_color=color, opacity=0.9,
                    text=[f"{val:.2f}"], textposition="outside",
                    width=0.4,
                    hovertemplate=f"<b>{group}</b><br>Avg {chosen_label}: %{{y:.2f}}<extra></extra>",
                ))
            fig_avg.update_layout(
                showlegend=False,
                title=" ",
                xaxis=dict(title="", gridcolor=GRID_COLOR),
                yaxis=dict(title=chosen_label, gridcolor=GRID_COLOR),
                barmode="group",
            )
            st.plotly_chart(style_fig(fig_avg, height=300), use_container_width=True)

        with c2:
            section_label("Distribution Range")
            fig_box = go.Figure()
            for label, vals, color in [("Churn", churn_vals, "#f87171"), ("Retained", retained_vals, "#34d399")]:
                fig_box.add_trace(go.Box(
                    y=vals, name=label,
                    marker_color=color, line_color=color,
                    boxpoints=False,
                    hovertemplate=f"<b>{label}</b><br>%{{y:.2f}}<extra></extra>",
                ))
            fig_box.update_layout(
                title=" ",
                yaxis=dict(title=chosen_label, gridcolor=GRID_COLOR),
                xaxis=dict(title=""),
                showlegend=False,
            )
            st.plotly_chart(style_fig(fig_box, height=300), use_container_width=True)


    # ═══════════════════════════════════════════════════════════════════
    # TAB 3 — PREDICT CHURN
    # ═══════════════════════════════════════════════════════════════════
    with tab3:
        section_label("Churn Prediction — User Input Data")

        if churn_model is None:
            st.error("Model prediksi tidak berhasil dimuat.")
        else:
            st.markdown("""<div style="font-size:13px;color:#9ca3af;margin-bottom:16px;">
                    Masukkan data user baru untuk diprediksi segmennya menggunakan model Light Gradient Boost Method.
                </div>""", unsafe_allow_html=True)
            
            pa, pb = st.columns(2)
            CHURN_PRESETS = {
                "Churn Potentional": {
                    "total_reviews": 2, "avg_stars": 3.3, "total_useful": 0,
                    "total_funny": 0, "total_cool": 0, "avg_sentiment_score": 0.71,
                    "avg_review_length": 633.0, "days_since_last_review": 1381,
                    "days_since_first_review": 1587, "votes_per_review": 2.0,
                },
                "Retained Potentional": {
                    "total_reviews": 15, "avg_stars": 4.2, "total_useful": 82,
                    "total_funny": 34, "total_cool": 33, "avg_sentiment_score": 0.52,
                    "avg_review_length": 435.0, "days_since_last_review": 767,
                    "days_since_first_review": 1009, "votes_per_review": 9.4,
                },
            }
            for col, (preset_name, preset_vals) in zip([pa, pb], CHURN_PRESETS.items()):
                with col:
                    if st.button(preset_name, key=f"preset_{preset_name}", use_container_width=True):
                        for feat, val in preset_vals.items():
                            st.session_state[f"churn_inp_{feat}"] = val
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            inputs = {}
            cols3  = st.columns(3)
            for i, feat in enumerate(avail_features):
                with cols3[i % 3]:
                    mn, mx, dv, step = INPUT_DEFAULTS.get(feat, (0.0, 100.0, 1.0, 0.1))
                    if f"churn_inp_{feat}" not in st.session_state:
                        st.session_state[f"churn_inp_{feat}"] = float(dv)
                    inputs[feat] = st.number_input(
                        FEATURE_LABELS.get(feat, feat),
                        help=FEATURE_DESCRIPTIONS.get(feat, ""),
                        min_value=float(mn), max_value=float(mx),
                        step=float(step), key=f"churn_inp_{feat}",
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Prediksi", use_container_width=True):
                try:
                    input_df = pd.DataFrame([[inputs[f] for f in avail_features]], columns=avail_features)
                    proba    = churn_model.predict_proba(input_df)[0]
                    pred     = int(churn_model.predict(input_df)[0])

                    churn_prob    = round(proba[1] * 100, 1)
                    retained_prob = round(proba[0] * 100, 1)
                    result_label  = "Churn" if pred == 1 else "Retained"
                    result_color  = CHURN_COLORS[result_label]
                    result_text   = "Churn" if pred == 1 else "Retained"

                    r1, r2, r3 = st.columns(3)
                    with r1: kpi_card("Prediction Results",       result_text,         "", result_color, border=True)
                    with r2: kpi_card("Churn Probability",    f"{churn_prob}%",    "", "#f87171")
                    with r3: kpi_card("Survival Probability", f"{retained_prob}%", "", "#34d399")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:#2e3250;border-radius:999px;height:12px;">
                        <div style="background:linear-gradient(to right,
                                    #34d399 {retained_prob}%, #f87171 {retained_prob}%);
                                    height:12px;border-radius:999px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:11px;color:#6b7280;margin-top:6px;">
                        <span>Retained {retained_prob}%</span>
                        <span>Churn {churn_prob}%</span>
                    </div>""", unsafe_allow_html=True)

                    divider()
                    section_label("Input vs Avg Group")

                    avg_churn    = df[df["churn"] == 1][avail_features].mean()
                    avg_retained = df[df["churn"] == 0][avail_features].mean()

                    EXCLUDE_CHART = {"days_since_first_review", "total_useful", "total_funny", "total_cool"}
                    chart_feats   = [f for f in avail_features if f not in EXCLUDE_CHART]

                    for i in range(0, len(chart_feats), 3):
                        cols = st.columns(3)
                        for col, feat in zip(cols, chart_feats[i:i+3]):
                            user_val  = round(inputs[feat], 2)
                            avg_c_val = round(float(avg_churn[feat]), 2)
                            avg_r_val = round(float(avg_retained[feat]), 2)
                            label     = FEATURE_LABELS.get(feat, feat)

                            dist_to_churn    = abs(user_val - avg_c_val)
                            dist_to_retained = abs(user_val - avg_r_val)
                            closer      = "churn" if dist_to_churn < dist_to_retained else "retained"
                            badge_color = "#f87171" if closer == "churn" else "#34d399"
                            badge_text  = "Mirip Churn" if closer == "churn" else "Mirip Retained"

                            lo, hi    = min(avg_c_val, avg_r_val), max(avg_c_val, avg_r_val)
                            span      = hi - lo if hi != lo else 1
                            pct_input = max(0, min(100, (user_val - lo) / span * 100))

                            with col:
                                st.markdown(f"""
                                <div style="padding:14px 16px;border-radius:12px;background:#1e2130;
                                            border:1px solid #2e3250;margin-bottom:10px;">
                                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                                        <span style="font-size:12px;font-weight:600;color:#e5e7eb;">{label}</span>
                                        <span style="font-size:10px;padding:2px 8px;border-radius:999px;
                                                     background:{badge_color}22;color:{badge_color};
                                                     border:1px solid {badge_color};">{badge_text}</span>
                                    </div>
                                    <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                                        <div style="text-align:center;">
                                            <div style="font-size:9px;color:#f87171;margin-bottom:2px;">AVG CHURN</div>
                                            <div style="font-size:14px;font-weight:600;color:#f87171;">{avg_c_val}</div>
                                        </div>
                                        <div style="text-align:center;">
                                            <div style="font-size:9px;color:{result_color};margin-bottom:2px;">YOUR INPUT</div>
                                            <div style="font-size:16px;font-weight:700;color:{result_color};">{user_val}</div>
                                        </div>
                                        <div style="text-align:center;">
                                            <div style="font-size:9px;color:#34d399;margin-bottom:2px;">AVG RETAINED</div>
                                            <div style="font-size:14px;font-weight:600;color:#34d399;">{avg_r_val}</div>
                                        </div>
                                    </div>
                                    <div style="position:relative;background:#2e3250;border-radius:999px;height:4px;">
                                        <div style="position:absolute;left:0;width:6px;height:6px;border-radius:50%;
                                                    background:#f87171;top:-1px;"></div>
                                        <div style="position:absolute;right:0;width:6px;height:6px;border-radius:50%;
                                                    background:#34d399;top:-1px;"></div>
                                        <div style="position:absolute;left:{pct_input:.0f}%;transform:translateX(-50%);
                                                    width:10px;height:10px;border-radius:50%;
                                                    background:{result_color};top:-3px;
                                                    box-shadow:0 0 6px {result_color};"></div>
                                    </div>
                                    <div style="display:flex;justify-content:space-between;
                                                font-size:9px;color:#4b5563;margin-top:6px;">
                                        <span>Churn end</span>
                                        <span>Retained end</span>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediksi gagal: {e}")