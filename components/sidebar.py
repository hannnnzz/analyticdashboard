import streamlit as st

def render_sidebar(df_business, df_exploded, df_review=None, df_user=None, df_checkin=None, df_wordcloud=None):
    st.sidebar.title("Filters")
    st.sidebar.caption("Filter berlaku untuk semua section.")

    df  = df_business.copy()
    dfx = df_exploded.copy()
    dfr = df_review.copy() if df_review is not None else None
    dfu = df_user.copy() if df_user is not None else None
    dfc = df_checkin.copy() if df_checkin is not None else None
    dfw = df_wordcloud.copy() if df_wordcloud is not None else None

    current_page = st.session_state.get("page", "Market Overview")

    # ── State ────────────────────────────────────────────────────────────
    if "state" in df.columns:
        states = sorted(df["state"].dropna().unique())
        all_states = st.sidebar.checkbox("Select all states", value=True)
        selected = states if all_states else st.sidebar.multiselect("States", states, default=states[:1])
        if selected:
            df  = df[df["state"].isin(selected)]
            dfx = dfx[dfx["state"].isin(selected)]

    # ── City ─────────────────────────────────────────────────────────────
    if "city" in df.columns:
        cities = sorted(df["city"].dropna().unique())
        all_cities = st.sidebar.checkbox("Select all cities", value=True)
        selected_c = cities if all_cities else st.sidebar.multiselect("Cities", cities, default=cities[:1])
        if selected_c:
            df  = df[df["city"].isin(selected_c)]
            dfx = dfx[dfx["city"].isin(selected_c)]

    # ── Price Range ───────────────────────────────────────────────────────────────
    if "RestaurantsPriceRange2_num" in df.columns:
        price_map = {1: "$ Budget", 2: "$$ Mid-Range", 3: "$$$ Upscale", 4: "$$$$ Fine Dining"}
        all_tiers = sorted(df["RestaurantsPriceRange2_num"].dropna().astype(int).unique())
        tier_labels = [price_map.get(t, str(t)) for t in all_tiers]
        all_price = st.sidebar.checkbox("Select all price ranges", value=True)
        selected_tiers = tier_labels if all_price else st.sidebar.multiselect("Price Range", options=tier_labels, default=tier_labels[:1])
        reverse_map = {v: k for k, v in price_map.items()}
        selected_tier_nums = [reverse_map[l] for l in selected_tiers if l in reverse_map]
        
        # hanya filter jika TIDAK select all
        if not all_price and selected_tier_nums:
            df  = df[df["RestaurantsPriceRange2_num"].isin(selected_tier_nums)]
            dfx = dfx[dfx["business_id"].isin(df["business_id"])]

    st.sidebar.caption("Pilih dahulu kategori untuk memperinci Nama Bisnis.")

    # ── Categories ────────────────────────────────────────────────────────
    if "categories_exploded" in dfx.columns:
        all_cats = sorted(dfx["categories_exploded"].dropna().unique())
        selected_cats = st.sidebar.multiselect("Categories", options=all_cats, default=[])
        if selected_cats:
            biz_in_cat = dfx[dfx["categories_exploded"].isin(selected_cats)]["business_id"].unique()
            df  = df[df["business_id"].isin(biz_in_cat)]
            dfx = dfx[dfx["business_id"].isin(biz_in_cat)]

        # ── Business Name (muncul hanya jika kategori dipilih) ────────────
        if selected_cats and "name" in df.columns:
            biz_names = sorted(df["name"].dropna().unique())
            all_biz = st.sidebar.checkbox("Select all businesses", value=True)
            selected_biz = (
                biz_names if all_biz
                else st.sidebar.multiselect("Business Name", options=biz_names, default=biz_names[:1])
            )
            if selected_biz:
                selected_biz_ids = df[df["name"].isin(selected_biz)]["business_id"].unique()
                df  = df[df["business_id"].isin(selected_biz_ids)]
                dfx = dfx[dfx["business_id"].isin(selected_biz_ids)]

    # ── Facilities ────────────────────────────────────────────────────────
    facility_map = {
        "BusinessAcceptsCreditCards": "Business ACC Credit Cards",
        "HasTV": "Has TV",
        "WheelchairAccessible": "Wheelchair Accessible",
        "RestaurantsTakeOut": "Restaurants Take Out",
        "WiFi_norm": "Wi-Fi",
        "RestaurantsReservations": "Restaurants Reservations",
        "GoodForKids": "Good For Kids",
        "RestaurantsDelivery": "Restaurants Delivery",
        "DogsAllowed": "Dogs Allowed",
        "RestaurantsGoodForGroups": "Rest Good For Groups",
        "OutdoorSeating": "Outdoor Seating",
        "DriveThru": "Drive Thru",
    }
    available_fac = {label: col for col, label in facility_map.items() if col in df.columns}
    if available_fac:
        selected_fac = st.sidebar.multiselect("Facilities", options=list(available_fac.keys()), default=[])
        if selected_fac:
            sel_cols = [available_fac[l] for l in selected_fac]
            mask = df[sel_cols].isin([True, "True", 1]).any(axis=1)
            df  = df[mask]
            dfx = dfx[dfx["business_id"].isin(df["business_id"])]

    # ── Ambience ──────────────────────────────────────────────────────────
    ambience_cols = [c for c in df.columns if c.startswith("Ambience_")]
    if ambience_cols:
        amb_labels = [c.replace("Ambience_", "").title() for c in ambience_cols]
        selected_amb = st.sidebar.multiselect("Ambience", options=amb_labels, default=[])
        if selected_amb:
            sel_cols = [f"Ambience_{a.lower()}" for a in selected_amb]
            mask = df[sel_cols].isin([True, "True", 1]).any(axis=1)
            df  = df[mask]
            dfx = dfx[dfx["business_id"].isin(df["business_id"])]

    # ── Parking ───────────────────────────────────────────────────────────
    parking_cols = [c for c in df.columns if c.startswith("BusinessParking_")]
    if parking_cols:
        park_labels = [c.replace("BusinessParking_", "").title() for c in parking_cols]
        selected_park = st.sidebar.multiselect("Parking Type", options=park_labels, default=[])
        if selected_park:
            sel_cols = [f"BusinessParking_{p.lower()}" for p in selected_park]
            mask = df[sel_cols].isin([True, "True", 1]).any(axis=1)
            df  = df[mask]
            dfx = dfx[dfx["business_id"].isin(df["business_id"])]

    # ── Music ─────────────────────────────────────────────────────────────
    music_cols = [c for c in df.columns if c.startswith("Music_")]
    if music_cols:
        music_labels = [c.replace("Music_", "").replace("_", " ").title() for c in music_cols]
        selected_music = st.sidebar.multiselect("Music Type", options=music_labels, default=[])
        if selected_music:
            sel_cols = [f"Music_{m.lower().replace(' ', '_')}" for m in selected_music]
            sel_cols = [c for c in sel_cols if c in df.columns]
            if sel_cols:
                mask = df[sel_cols].isin([True, "True", 1]).any(axis=1)
                df  = df[mask]
                dfx = dfx[dfx["business_id"].isin(df["business_id"])]

    # ── Best Nights ───────────────────────────────────────────────────────
    night_cols = [c for c in df.columns if c.startswith("BestNights_")]
    if night_cols:
        night_labels = [c.replace("BestNights_", "").title() for c in night_cols]
        selected_nights = st.sidebar.multiselect("Best Nights", options=night_labels, default=[])
        if selected_nights:
            sel_cols = [f"BestNights_{n.lower()}" for n in selected_nights]
            sel_cols = [c for c in sel_cols if c in df.columns]
            if sel_cols:
                mask = df[sel_cols].isin([True, "True", 1]).any(axis=1)
                df  = df[mask]
                dfx = dfx[dfx["business_id"].isin(df["business_id"])]

    # ── Propagate ke semua df via business_id / user_id ──────────────────
    filtered_biz_ids = set(df["business_id"].unique())

    if dfr is not None and "business_id" in dfr.columns:
        dfr = dfr[dfr["business_id"].isin(filtered_biz_ids)]

    if dfc is not None and "business_id" in dfc.columns:
        dfc = dfc[dfc["business_id"].isin(filtered_biz_ids)]

    if dfw is not None and "business_id" in dfw.columns:
        dfw = dfw[dfw["business_id"].isin(filtered_biz_ids)]

    if dfu is not None and "user_id" in dfu.columns and dfr is not None and "user_id" in dfr.columns:
        filtered_user_ids = set(dfr["user_id"].unique())
        dfu = dfu[dfu["user_id"].isin(filtered_user_ids)]

    return df, dfx, dfr, dfu, dfc, dfw