import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.style import apply_style, page_header, section_label, divider, style_fig, FONT_COLOR, GRID_COLOR

# ── Color palette ────────────────────────────────────────────────────────────
CHURN_COLORS = {
    "Overdue":  "#f87171",
    "Active":   "#34d399",
}

GROWTH_COLORS = {
    "stable":   "#60a5fa",
    "declining":"#f87171",
    "growing":  "#34d399",
    "early":    "#fbbf24",
    "dormant":  "#9ca3af",
}

FREQ_COLORS = {
    "loyal":      "#818cf8",
    "regular":    "#60a5fa",
    "occasional": "#fbbf24",
    "one-time":   "#9ca3af",
}

# ── Feature metadata ─────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "total_checkins":          "Total Checkins",
    "checkins_per_year":       "Checkins / Year",
    "avg_gap_days":            "Avg Gap Between Visits (Days)",
    "hazard_indicator":        "Hazard Indicator",
    "checkin_trend_slope":     "Checkin Trend Slope",
    "checkin_dropoff_ratio":   "Checkin Dropoff Ratio",
    "longest_inactive_streak": "Longest Inactive Streak (Days)",
    "monthly_checkins_mean":   "Monthly Checkins (Avg)",
    "pre_covid_checkins":      "Pre-COVID Checkins",
    "covid_checkins":          "COVID-Period Checkins",
    "post_covid_checkins":     "Post-COVID Checkins",
    "business_age_years":      "Business Age (Years)",
    "unique_months_active":    "Unique Months Active",
}

FEATURE_DESCRIPTIONS = {
    "total_checkins":          "Total number of checkins recorded for the business",
    "checkins_per_year":       "Average number of checkins per year",
    "avg_gap_days":            "Average number of days between consecutive visits",
    "hazard_indicator":        "Risk score — higher value means higher churn risk",
    "checkin_trend_slope":     "Direction of checkin trend over time (negative = declining)",
    "checkin_dropoff_ratio":   "Ratio of recent checkins vs previous period (< 1 = drop)",
    "longest_inactive_streak": "Longest period (days) with no checkin activity",
    "monthly_checkins_mean":   "Average number of checkins per active month",
    "pre_covid_checkins":      "Total checkins before COVID-19 period",
    "covid_checkins":          "Total checkins during COVID-19 period",
    "post_covid_checkins":     "Total checkins after COVID-19 period",
    "business_age_years":      "How many years the business has been active",
    "unique_months_active":    "Number of distinct months the business had at least one checkin",
}

PREDICT_FEATURES = [
    "total_checkins", "checkins_per_year", "avg_gap_days",
    "hazard_indicator", "checkin_trend_slope", "checkin_dropoff_ratio",
    "longest_inactive_streak", "monthly_checkins_mean",
    "pre_covid_checkins", "covid_checkins", "post_covid_checkins",
    "business_age_years", "unique_months_active",
]

INPUT_DEFAULTS = {
    "total_checkins":          (0,    5000,  100,   1),
    "checkins_per_year":       (0.0,  5000.0,100.0, 1.0),
    "avg_gap_days":            (0.0,  2000.0,120.0, 1.0),
    "hazard_indicator":        (-250.0, 5200.0, 5.0, 0.1),
    "checkin_trend_slope":     (-3.5, 1.5,   -0.01, 0.001),
    "checkin_dropoff_ratio":   (0.0,  34.0,  1.0,  0.01),
    "longest_inactive_streak": (0,    5000,  400,  1),
    "monthly_checkins_mean":   (0.0,  5000.0,100.0, 1.0),
    "pre_covid_checkins":      (0,    5000,  100,  1),
    "covid_checkins":          (0,    1000,  7,    1),
    "post_covid_checkins":     (0,    500,   0,    1),
    "business_age_years":      (0.0,  30.0,  5.0,  0.1),
    "unique_months_active":    (0,    200,   50,   1),
}


# ── Helper components ─────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", color="#60a5fa"):
    fs = "18px" if len(str(value)) > 10 else "24px"
    sub_html = f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="padding:16px;border-radius:10px;background:#1e2130;border:1px solid #2e3250;
                text-align:center;min-height:120px;display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
        <div style="font-size:{fs};font-weight:700;color:{color};">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


# ── Main render ───────────────────────────────────────────────────────────────
def render(df_checkin, churn_model=None):
    apply_style()
    page_header("Business Churn — Checkin Behavior", "Menganalisis pola aktivitas bisnis dan risiko churn berdasarkan data check-in Yelp.")

    df = df_checkin.copy()

    if "overdue_visit_flag" not in df.columns:
        st.warning("Column 'overdue_visit_flag' not found in dataset.")
        return

    df["status_label"] = df["overdue_visit_flag"].map({1: "Overdue", 0: "Active"})

    total      = len(df)
    n_overdue  = int(df["overdue_visit_flag"].sum())
    n_active   = total - n_overdue
    overdue_rate = n_overdue / total * 100

    # ── KPI ───────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total Businesses", f"{total:,}",       "in dataset",      "#60a5fa")
    with k2: kpi_card("Overdue",          f"{n_overdue:,}",   "no recent visit",  "#f87171")
    with k3: kpi_card("Active",           f"{n_active:,}",    "still visiting",   "#34d399")
    with k4: kpi_card("Overdue Rate",     f"{overdue_rate:.1f}%", "of all businesses", "#fbbf24")

    divider()

    tab1, tab2 = st.tabs(["Overview", "Feature Analysis"])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        c1, c2 = st.columns([1, 1.7])

        with c1:
            section_label("Business Status")
            seg = df["status_label"].value_counts().reset_index()
            seg.columns = ["label", "count"]
            fig_pie = go.Figure(go.Pie(
                labels=seg["label"], values=seg["count"], hole=0.62,
                marker=dict(colors=[CHURN_COLORS.get(s, "#9ca3af") for s in seg["label"]], line=dict(width=0)),
                textinfo="percent+label",
                textfont=dict(color="#e5e7eb", size=13),
                hovertemplate="<b>%{label}</b><br>%{value:,} businesses<br>%{percent}<extra></extra>",
                direction="clockwise", sort=False,
            ))
            fig_pie.update_layout(
                showlegend=False, title=" ",
                annotations=[dict(
                    text=f"<b>{overdue_rate:.1f}%</b><br>Overdue",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color="#f87171", size=20),
                )],
            )
            st.plotly_chart(style_fig(fig_pie, height=320), use_container_width=True)

        with c2:
            section_label("Growth Phase Distribution")
            if "growth_phase" in df.columns:
                # count per growth_phase x status
                gp = df.groupby(["growth_phase", "status_label"]).size().reset_index(name="count")

                # create a title-cased label for display (e.g. "dormant" -> "Dormant")
                gp["growth_phase_title"] = gp["growth_phase"].astype(str).str.title()

                # keep a stable x-order based on first appearance
                phases = gp.drop_duplicates(subset=["growth_phase"])["growth_phase_title"].tolist()

                gp["color"] = gp["growth_phase"].map(lambda x: GROWTH_COLORS.get(x, "#9ca3af"))
                fig_gp = go.Figure()
                for status, color in [("Overdue", "#f87171"), ("Active", "#34d399")]:
                    sub = gp[gp["status_label"] == status]
                    fig_gp.add_trace(go.Bar(
                        name=status,
                        x=sub["growth_phase_title"],    # <-- use title-cased labels here
                        y=sub["count"],
                        marker_color=color, opacity=0.85,
                        text=sub["count"], textposition="outside",
                        hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>",
                    ))
                fig_gp.update_layout(
                    barmode="group", title=" ",
                    xaxis=dict(title="", gridcolor=GRID_COLOR,
                            categoryorder="array", categoryarray=phases),  # preserve order
                    yaxis=dict(title="Businesses", gridcolor=GRID_COLOR),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(style_fig(fig_gp, height=320), use_container_width=True)

        divider()

        c3, c4 = st.columns(2)

        with c3:
            section_label("Engagement Level — Active vs Overdue")

            if "visit_frequency_label" in df.columns:

                ENGAGEMENT_LABELS = {
                    "loyal":      "Highly Engaged Influencers",
                    "regular":    "Regular Reviewers",
                    "occasional": "Casual Users",
                    "one-time":   "Active Contributors",
                }

                vf = df.groupby(["visit_frequency_label", "status_label"]).size().reset_index(name="count")

                def _norm_key(x):
                    k = str(x).lower().strip()
                    k = k.replace("_", "-").replace(" ", "-")
                    return k

                vf["engagement_key"] = vf["visit_frequency_label"].map(lambda x: _norm_key(x))
                vf["engagement_label"] = vf["engagement_key"].map(lambda k: ENGAGEMENT_LABELS.get(k, k.title().replace("-", " ")))

                phases = vf.drop_duplicates(subset=["engagement_key"])["engagement_label"].tolist()
                fig_vf = go.Figure()

                for status, color in [("Overdue", "#f87171"), ("Active", "#34d399")]:
                    sub = vf[vf["status_label"] == status]

                    if phases:
                        sub = sub.set_index("engagement_label").reindex(phases).reset_index()
                        sub["count"] = sub["count"].fillna(0).astype(int)

                    fig_vf.add_trace(go.Bar(
                        name=status,
                        x=sub["engagement_label"],
                        y=sub["count"],
                        marker_color=color,
                        opacity=0.85,
                        text=sub["count"],
                        textposition="outside",
                        hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>",
                    ))

                fig_vf.update_layout(
                    barmode="group",
                    title=" ",
                    xaxis=dict(title="", gridcolor=GRID_COLOR, categoryorder="array", categoryarray=phases),
                    yaxis=dict(title="Businesses", gridcolor=GRID_COLOR),
                    legend=dict(
                        orientation="v",
                        x=1.02,   
                        y=1.0,
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                    ),
                    margin=dict(r=150)
                )

                st.plotly_chart(style_fig(fig_vf, height=300), use_container_width=True)

        with c4:
            section_label("Pre / COVID / Post Checkin Volume")
            covid_cols = ["pre_covid_checkins", "covid_checkins", "post_covid_checkins"]
            if all(c in df.columns for c in covid_cols):
                covid_avg = df.groupby("status_label")[covid_cols].mean().reset_index()
                labels_map = {
                    "pre_covid_checkins":  "Pre-COVID",
                    "covid_checkins":      "During COVID",
                    "post_covid_checkins": "Post-COVID",
                }
                fig_covid = go.Figure()
                for status, color in [("Overdue", "#f87171"), ("Active", "#34d399")]:
                    row = covid_avg[covid_avg["status_label"] == status]
                    if row.empty:
                        continue
                    vals = [float(row[c].values[0]) for c in covid_cols]
                    fig_covid.add_trace(go.Bar(
                        name=status,
                        x=[labels_map[c] for c in covid_cols],
                        y=vals,
                        marker_color=color, opacity=0.85,
                        text=[f"{v:.1f}" for v in vals], textposition="outside",
                        hovertemplate=f"<b>%{{x}}</b><br>Avg: %{{y:.1f}}<extra></extra>",
                    ))
                fig_covid.update_layout(
                    barmode="group", title=" ",
                    xaxis=dict(title="", gridcolor=GRID_COLOR),
                    yaxis=dict(title="Avg Checkins", gridcolor=GRID_COLOR),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(style_fig(fig_covid, height=300), use_container_width=True)


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — FEATURE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        section_label("What Separates Overdue from Active Businesses?")

        avail_features = [f for f in PREDICT_FEATURES if f in df.columns]
        feat_options   = {FEATURE_LABELS.get(f, f): f for f in avail_features}

        col_sel, col_desc = st.columns([1, 2])
        with col_sel:
            chosen_label = st.selectbox(
                "Select Feature", list(feat_options.keys()),
                key="checkin_tab2_feat", label_visibility="collapsed"
            )
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

        overdue_vals = df[df["overdue_visit_flag"] == 1][feat_dist].dropna()
        active_vals  = df[df["overdue_visit_flag"] == 0][feat_dist].dropna()
        avg_o = overdue_vals.mean()
        avg_a = active_vals.mean()
        diff  = avg_a - avg_o
        pct   = (abs(diff) / (abs(avg_o) + 1e-9)) * 100
        arrow = "▲" if diff > 0 else "▼"
        w_color = "#34d399" if diff > 0 else "#f87171"

        # Insight banner
        st.markdown(f"""
        <div style="padding:16px 24px;border-radius:12px;background:#1e2130;
                    border:1px solid #2e3250;margin-bottom:20px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:4px;">Key Insight</div>
                <div style="font-size:15px;color:#e5e7eb;font-weight:600;">
                    Active businesses have
                    <span style="color:{w_color};">{arrow} {abs(pct):.0f}% {"higher" if diff > 0 else "lower"}</span>
                    {chosen_label.lower()} compared to overdue ones.
                </div>
            </div>
            <div style="font-size:28px;">{"📈" if diff > 0 else "📉"}</div>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1: kpi_card("Avg — Overdue", f"{avg_o:.2f}", "overdue businesses", "#f87171")
        with k2: kpi_card("Avg — Active",  f"{avg_a:.2f}", "active businesses",  "#34d399")
        with k3: kpi_card("Difference",    f"{arrow} {abs(diff):.2f}", f"{pct:.0f}% gap", w_color)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.4, 1])
        with c1:
            section_label("Average Value Comparison")
            fig_avg = go.Figure()
            for group, val, color in [("Overdue", avg_o, "#f87171"), ("Active", avg_a, "#34d399")]:
                fig_avg.add_trace(go.Bar(
                    name=group, x=[group], y=[val],
                    marker_color=color, opacity=0.9,
                    text=[f"{val:.2f}"], textposition="outside",
                    width=0.4,
                    hovertemplate=f"<b>{group}</b><br>Avg {chosen_label}: %{{y:.2f}}<extra></extra>",
                ))
            fig_avg.update_layout(
                showlegend=False, title=" ",
                xaxis=dict(title="", gridcolor=GRID_COLOR),
                yaxis=dict(title=chosen_label, gridcolor=GRID_COLOR),
                barmode="group",
            )
            st.plotly_chart(style_fig(fig_avg, height=300), use_container_width=True)

        with c2:
            section_label("Distribution Range")
            fig_box = go.Figure()
            for label, vals, color in [("Overdue", overdue_vals, "#f87171"), ("Active", active_vals, "#34d399")]:
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
