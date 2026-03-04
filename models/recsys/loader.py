import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ── Path ──────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__))


# =============================================================================
# MODEL DEFINITION (harus sama persis dengan di notebook)
# =============================================================================
class HybridNCF(nn.Module):
    def __init__(self, n_users, n_items, n_features,
                 emb_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super().__init__()

        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        self.content_proj = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        layers = []
        in_dim = emb_dim * 2 + 64
        for out_dim in mlp_layers:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.output  = nn.Linear(emb_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, user, item, features):
        gmf_out = self.gmf_user(user) * self.gmf_item(item)
        content = self.content_proj(features)
        mlp_in  = torch.cat([self.mlp_user(user), self.mlp_item(item), content], dim=1)
        mlp_out = self.mlp(mlp_in)
        x = torch.cat([gmf_out, mlp_out], dim=1)
        return self.sigmoid(self.output(x)).squeeze()


# =============================================================================
# LOAD FUNCTIONS (dipanggil dari app.py)
# =============================================================================
def load_ncf_model():
    """Load Hybrid NCF model dari file .pt"""
    try:
        config_path = os.path.join(MODEL_DIR, "model_config.pkl")
        model_path  = os.path.join(MODEL_DIR, "hybrid_ncf_model.pt")

        with open(config_path, "rb") as f:
            cfg = pickle.load(f)

        model = HybridNCF(
            n_users    = cfg["n_users"],
            n_items    = cfg["n_items"],
            n_features = cfg["n_features"],
            emb_dim    = cfg.get("emb_dim", 32),
            mlp_layers = cfg.get("mlp_layers", [256, 128, 64]),
            dropout    = cfg.get("dropout", 0.3),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[load_ncf_model] Error: {e}")
        return None


def load_label_encoder_user():
    try:
        with open(os.path.join(MODEL_DIR, "le_user.pkl"), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[load_label_encoder_user] Error: {e}")
        return None


def load_label_encoder_biz():
    try:
        with open(os.path.join(MODEL_DIR, "le_biz.pkl"), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[load_label_encoder_biz] Error: {e}")
        return None


def load_biz_features():
    """Load BizLookup — DataFrame berisi feature_cols + biz_idx per business"""
    try:
        return pd.read_pickle(os.path.join(MODEL_DIR, "biz_lookup.pkl"))
    except Exception as e:
        print(f"[load_biz_features] Error: {e}")
        return None


def load_biz_info():
    try:
        biz_info = pd.read_pickle(os.path.join(MODEL_DIR, "biz_info.pkl"))
        
        # Enrich name & city dari loader_business
        from data.loader_business import load_business
        biz_excel = load_business()[['business_id', 'name', 'city']]
        biz_info  = biz_info.merge(biz_excel, on='business_id', how='left', suffixes=('', '_excel'))
        
        for col in ['name', 'city']:
            col_excel = col + '_excel'
            if col_excel in biz_info.columns:
                if col in biz_info.columns:
                    biz_info[col] = biz_info[col].fillna(biz_info[col_excel])
                else:
                    biz_info[col] = biz_info[col_excel]
                biz_info = biz_info.drop(columns=[col_excel])
        
        return biz_info
    except Exception as e:
        print(f"[load_biz_info] Error: {e}")
        return None


def load_df_slim():
    """Load df_slim — user_id, user_idx, biz_idx, stars + user features"""
    try:
        return pd.read_pickle(os.path.join(MODEL_DIR, "df_slim.pkl"))
    except Exception as e:
        print(f"[load_df_slim] Error: {e}")
        return None


# =============================================================================
# INFERENCE FUNCTIONS (dipanggil dari views/recsys.py)
# =============================================================================
def get_user_history(df_slim, le_biz, biz_info, user_id):
    """Ambil histori review user, di-enrich dengan biz_info."""
    if df_slim is None or le_biz is None:
        return pd.DataFrame()

    user_rows = df_slim[df_slim["user_id"] == user_id].copy()
    if user_rows.empty:
        return pd.DataFrame()

    try:
        user_rows["business_id"] = le_biz.inverse_transform(user_rows["biz_idx"].astype(int))
    except Exception:
        return pd.DataFrame()

    if biz_info is not None:
        user_rows = user_rows.merge(
            biz_info[["business_id", "name", "categories", "city", "stars"]],
            on="business_id", how="left", suffixes=("_review", "")
        )
        if "stars_review" in user_rows.columns:
            user_rows["stars"] = user_rows["stars"].fillna(user_rows["stars_review"])
            user_rows = user_rows.drop(columns=["stars_review"], errors="ignore")

    if biz_info is not None and "mapped_category" in biz_info.columns:
        mc = biz_info[["business_id", "mapped_category"]].drop_duplicates("business_id")
        user_rows = user_rows.merge(mc, on="business_id", how="left")

    return user_rows.reset_index(drop=True)


def get_recommendations(
    model, le_user, le_biz, biz_features, biz_info,
    df_slim, user_id, top_n=10, exclude_seen=True,
    max_per_category=2, device=None
):
    """Generate top-N personalized recommendations untuk satu user."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    if user_id not in le_user.classes_:
        return pd.DataFrame()

    user_idx = le_user.transform([user_id])[0]

    # Bisnis yang sudah di-review
    reviewed_biz_idx = set()
    if exclude_seen and df_slim is not None:
        reviewed_biz_idx = set(
            df_slim[df_slim["user_id"] == user_id]["biz_idx"].astype(int).values
        )

    # Profil user asli dari df_slim
    user_profile = {}
    if df_slim is not None:
        user_rows = df_slim[df_slim["user_id"] == user_id]
        if not user_rows.empty:
            for col in ["user_review_count", "user_avg_stars", "user_fans",
                        "user_is_elite", "user_friend_count"]:
                if col in user_rows.columns:
                    user_profile[col] = float(user_rows[col].iloc[0])

    # Load feature_cols dari pickle (paling aman, urutan kolom terjamin)
    feat_cols_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    if os.path.exists(feat_cols_path):
        with open(feat_cols_path, "rb") as f:
            feature_cols = pickle.load(f)
        feature_cols = [c for c in feature_cols if c in biz_features.columns]
    else:
        # Fallback: exclude kolom metadata
        exclude_meta = {
            "business_id", "user_id", "biz_idx", "user_idx", "stars",
            "sample_weight", "mapped_category", "name", "categories", "city"
        }
        feature_cols = [
            c for c in biz_features.columns
            if c not in exclude_meta
            and str(biz_features[c].dtype) in ["float64", "float32", "int64", "int32", "bool"]
        ]

    # Kandidat bisnis (exclude yang sudah dilihat)
    candidates = biz_features[
        ~biz_features["biz_idx"].isin(reviewed_biz_idx)
    ].reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame()

    # Build feature matrix dengan profil user ASLI
    feat_df = candidates[feature_cols].fillna(0).copy()
    for col, val in user_profile.items():
        if col in feat_df.columns:
            feat_df[col] = val
    for col in ["signed_sentiment", "recency_weight"]:
        if col in feat_df.columns:
            feat_df[col] = 0.0

    # Inference dalam batch biar tidak OOM
    BATCH = 2048
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(candidates), BATCH):
            end    = min(start + BATCH, len(candidates))
            u_t    = torch.LongTensor([user_idx] * (end - start)).to(device)
            i_t    = torch.LongTensor(candidates["biz_idx"].values[start:end]).to(device)
            f_t    = torch.FloatTensor(feat_df.iloc[start:end].values).to(device)
            scores = model(u_t, i_t, f_t).cpu().numpy()
            all_scores.extend(scores)

    all_scores = np.array(all_scores)
    
    # Hitung kategori & kota favorit user dari histori review
    user_history = df_slim[df_slim["user_id"] == user_id] if df_slim is not None else pd.DataFrame()
    
    fav_cats   = set()
    fav_cities = set()
    
    if not user_history.empty and biz_info is not None:
        try:
            hist_biz_ids = le_biz.inverse_transform(user_history["biz_idx"].astype(int))
            hist_info    = biz_info[biz_info["business_id"].isin(hist_biz_ids)]
            
            # Kategori favorit = yang sering di-review dengan rating tinggi
            high_rated = user_history[user_history["stars"] >= 4]
            if not high_rated.empty:
                high_biz_ids = le_biz.inverse_transform(high_rated["biz_idx"].astype(int))
                high_info    = biz_info[biz_info["business_id"].isin(high_biz_ids)]
                if "mapped_category" in high_info.columns:
                    fav_cats = set(high_info["mapped_category"].dropna().unique())
                if "city" in high_info.columns:
                    fav_cities = set(high_info["city"].dropna().unique())
        except Exception:
            pass

    # Boost score untuk bisnis yang cocok dengan preferensi user
    all_scores_boosted = all_scores.copy()
    if fav_cats or fav_cities:
        for idx_c, biz_idx_val in enumerate(candidates["biz_idx"].values):
            biz_id = le_biz.inverse_transform([int(biz_idx_val)])[0]
            biz_row = biz_info[biz_info["business_id"] == biz_id] if biz_info is not None else pd.DataFrame()
            if biz_row.empty:
                continue
            boost = 0.0
            if fav_cats and "mapped_category" in biz_row.columns:
                if biz_row["mapped_category"].iloc[0] in fav_cats:
                    boost += 0.05   # +5% untuk kategori favorit
            if fav_cities and "city" in biz_row.columns:
                if biz_row["city"].iloc[0] in fav_cities:
                    boost += 0.03   # +3% untuk kota favorit
            all_scores_boosted[idx_c] = min(all_scores_boosted[idx_c] + boost, 1.0)
    
    all_scores = all_scores_boosted

    # Top N*3 untuk diversity filter
    top_idx = np.argsort(all_scores)[::-1][:top_n * 3]
    result  = candidates.iloc[top_idx].copy()
    result["predicted_score"] = all_scores[top_idx]
    result["business_id"]     = le_biz.inverse_transform(result["biz_idx"].astype(int))

    # Enrich dengan biz_info
    if biz_info is not None:
        merge_cols = ["business_id"] + [
            c for c in ["name", "categories", "city", "stars", "mapped_category"]
            if c in biz_info.columns
        ]
        result = result.merge(
            biz_info[merge_cols].drop_duplicates("business_id"),
            on="business_id", how="left"
        )

    # Diversity: max max_per_category per kategori
    cat_col = "mapped_category" if "mapped_category" in result.columns else "categories"
    seen_cats, diverse_rows = {}, []
    for _, row in result.iterrows():
        cat = str(row.get(cat_col, "Other"))
        if seen_cats.get(cat, 0) < max_per_category:
            diverse_rows.append(row)
            seen_cats[cat] = seen_cats.get(cat, 0) + 1
        if len(diverse_rows) >= top_n:
            break

    result = pd.DataFrame(diverse_rows).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result