# app.py — Real Estate House Price Prediction (Streamlit)
# Loads saved artifacts (model, scaler, encoder) and serves predictions + EDA + importances.

import os
import io
import pickle
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------- Page Config & Style -------------------------
st.set_page_config(page_title="Real Estate House Price Prediction", page_icon="🏠", layout="wide")
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .metric-label { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Artifact Paths -------------------------
ART_DIR = os.getenv("MODEL_DIR", "model")
MODEL_PATH  = os.path.join(ART_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
ENC_PATH    = os.path.join(ART_DIR, "onehot_encoder.pkl")
TRAIN_COLS  = os.path.join(ART_DIR, "training_columns.pkl")
NUM_COLS    = os.path.join(ART_DIR, "num_cols.pkl")
CAT_COLS    = os.path.join(ART_DIR, "cat_cols.pkl")

# ------------------------- Load Artifacts -------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(TRAIN_COLS, "rb") as f:
        training_columns = pickle.load(f)
    with open(NUM_COLS, "rb") as f:
        num_cols = pickle.load(f)
    with open(CAT_COLS, "rb") as f:
        cat_cols = pickle.load(f)

    try:
        cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    except Exception:
        try:
            cat_feature_names = list(encoder.get_feature_names(cat_cols))
        except Exception:
            # last-resort fallback
            cat_feature_names = [f"cat_{i}" for i in range(getattr(encoder, "categories_", [[]]).__len__())]

    ohe_feature_names = list(num_cols) + cat_feature_names
    return model, scaler, encoder, training_columns, num_cols, cat_cols, ohe_feature_names

# ------------------------- Preprocessing Helpers -------------------------
def context_aware_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    none_fill = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
                 "GarageType","GarageFinish","GarageQual","GarageCond",
                 "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
                 "MasVnrType"]
    zero_fill = ["GarageYrBlt","GarageArea","GarageCars",
                 "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
                 "BsmtFullBath","BsmtHalfBath","MasVnrArea"]
    for c in none_fill:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    for c in zero_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda s: s.fillna(s.median()))

    for c in df.select_dtypes(include="object").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def has(cols: List[str]) -> bool: return all([c in df.columns for c in cols])

    if has(["TotalBsmtSF","1stFlrSF","2ndFlrSF"]):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    for c in ["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalBath"] = df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]

    if has(["YrSold","YearBuilt"]):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    if has(["YrSold","YearRemodAdd"]):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    if has(["YrSold","YearBuilt"]):
        df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    return df

# ------------------------- Strict Transform & Predict -------------------------
def transform_for_model(df: pd.DataFrame, scaler, encoder, training_columns, num_cols, cat_cols) -> np.ndarray:
    # Align to training schema
    df = df.reindex(columns=training_columns)

    # Numeric block -> float64, coerce errors, fill NaNs, sanitize inf
    if len(num_cols) > 0:
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
        num_df = num_df.fillna(0.0).astype("float64")
        num_arr = num_df.to_numpy(dtype="float64", copy=False)
        num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        num_arr = np.empty((len(df), 0), dtype="float64")

    # Categorical block -> str, replace NaNs to "None"
    if len(cat_cols) > 0:
        cat_df = df[cat_cols].astype("object").copy()
        for c in cat_df.columns:
            cat_df[c] = cat_df[c].astype(str)
            cat_df[c] = cat_df[c].replace({"nan": "None", "NaN": "None", "None": "None"}).fillna("None")
        cat_arr = cat_df.to_numpy(dtype="object", copy=False)
    else:
        cat_arr = np.empty((len(df), 0), dtype="object")

    # Transform
    X_num = scaler.transform(num_arr) if num_arr.shape[1] else np.empty((len(df), 0))
    X_cat = encoder.transform(cat_arr) if cat_arr.shape[1] else np.empty((len(df), 0))
    if hasattr(X_cat, "toarray"):  # handle sparse
        X_cat = X_cat.toarray()

    return np.hstack([X_num, X_cat]) if X_cat.size else X_num

def predict_prices(df: pd.DataFrame) -> np.ndarray:
    model, scaler, encoder, training_columns, num_cols, cat_cols, _ = load_artifacts()
    df2 = context_aware_impute(df.copy())
    df2 = feature_engineer(df2)
    X = transform_for_model(df2, scaler, encoder, training_columns, num_cols, cat_cols)
    y_log = model.predict(X)
    return np.expm1(y_log)

# ------------------------- Header -------------------------
c1, c2 = st.columns([1, 2])
with c1:
    st.title("🏠 Real Estate House Price Prediction")
with c2:
    st.caption("Predict house prices using a trained ML model with preprocessing, feature engineering, and real-time analysis.")

# ------------------------- Sidebar (Artifact Status) -------------------------
st.sidebar.header("Model Files Status")
missing = [p for p in [MODEL_PATH, SCALER_PATH, ENC_PATH, TRAIN_COLS, NUM_COLS, CAT_COLS] if not os.path.exists(p)]
if missing:
    st.sidebar.error("Missing files:\n" + "\n".join([f"- {os.path.relpath(m)}" for m in missing]))
else:
    st.sidebar.success(f"✅ All model artifacts found in: `{ART_DIR}`")

# ------------------------- Tabs -------------------------
tab_predict, tab_batch, tab_eda, tab_importance = st.tabs(
    ["🔮 Predict (Form)", "📦 Batch Prediction", "📊 Data EDA", "⭐ Feature Importance"]
)

# ------------------------- Predict (Form) -------------------------
with tab_predict:
    st.subheader("Single Prediction")
    left, right = st.columns([1,1])

    with left:
        GrLivArea = st.number_input("GrLivArea (sqft)", min_value=100, max_value=8000, value=1500, step=50)
        TotalBsmtSF = st.number_input("TotalBsmtSF (sqft)", min_value=0, max_value=4000, value=800, step=50)
        F1 = st.number_input("1stFlrSF (sqft)", min_value=0, max_value=4000, value=900, step=50)
        F2 = st.number_input("2ndFlrSF (sqft)", min_value=0, max_value=3000, value=600, step=50)

    with right:
        FullBath = st.number_input("FullBath", min_value=0, max_value=5, value=2, step=1)
        HalfBath = st.number_input("HalfBath", min_value=0, max_value=5, value=1, step=1)
        BsmtFullBath = st.number_input("BsmtFullBath", min_value=0, max_value=3, value=0, step=1)
        BsmtHalfBath = st.number_input("BsmtHalfBath", min_value=0, max_value=3, value=0, step=1)

    col3, col4, col5 = st.columns(3)
    with col3:
        OverallQual = st.slider("OverallQual (1-10)", 1, 10, 6)
    with col4:
        GarageCars = st.slider("GarageCars", 0, 5, 2)
    with col5:
        YrSold = st.number_input("YrSold", min_value=2006, max_value=2010, value=2008)

    col6, col7 = st.columns(2)
    with col6:
        YearBuilt  = st.number_input("YearBuilt", min_value=1870, max_value=2025, value=2003)
    with col7:
        Neighborhood = st.text_input("Neighborhood", "NAmes")
    ExterQual   = st.selectbox("ExterQual", ["Ex","Gd","TA","Fa","Po","None"], index=2)

    row = pd.DataFrame({
        "GrLivArea":[GrLivArea], "TotalBsmtSF":[TotalBsmtSF], "1stFlrSF":[F1], "2ndFlrSF":[F2],
        "FullBath":[FullBath], "HalfBath":[HalfBath], "BsmtFullBath":[BsmtFullBath], "BsmtHalfBath":[BsmtHalfBath],
        "OverallQual":[OverallQual], "GarageCars":[GarageCars], "YearBuilt":[YearBuilt], "YrSold":[YrSold],
        "Neighborhood":[Neighborhood], "ExterQual":[ExterQual]
    })

    if st.button("Predict Price"):
        try:
            pred = predict_prices(row)[0]
            st.metric("Predicted SalePrice ($)", f"{pred:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------- Batch Predict (CSV) -------------------------
with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV with house features (no `SalePrice`). If it contains an `Id` column, it will be preserved.")
    up = st.file_uploader("Upload features CSV", type=["csv"], key="batch_up")
    if up is not None:
        try:
            df = pd.read_csv(up)
            idx = df["Id"] if "Id" in df.columns else pd.Series(np.arange(1, len(df)+1), name="Id")
            preds = predict_prices(df)
            out = pd.DataFrame({"Id": idx, "SalePrice": preds})
            st.write(out.head())

            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button("Download predictions.csv", data=buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ------------------------- EDA -------------------------
with tab_eda:
    st.subheader("Quick EDA")
    st.markdown("Upload a CSV to visualize distributions and correlations of numeric features.")
    eda_up = st.file_uploader("Upload CSV for EDA", type=["csv"], key="eda_up")
    if eda_up is not None:
        try:
            df = pd.read_csv(eda_up)
            st.write("Preview:", df.head())

            num_df = df.select_dtypes(include=[np.number])
            if num_df.shape[1] > 0:
                ncols = 3
                cols = st.columns(ncols)
                for i, col in enumerate(num_df.columns[:9]):
                    with cols[i % ncols]:
                        fig, ax = plt.subplots(figsize=(3.5, 3))
                        ax.hist(num_df[col].dropna().values, bins=30)
                        ax.set_title(col)
                        st.pyplot(fig, clear_figure=True)

                if num_df.shape[1] >= 2:
                    corr = num_df.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(6.5, 5))
                    cax = ax.imshow(corr.values, aspect="auto")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=90)
                    ax.set_yticklabels(corr.columns)
                    fig.colorbar(cax)
                    ax.set_title("Correlation Heatmap")
                    st.pyplot(fig, clear_figure=True)
            else:
                st.info("No numeric columns found for EDA.")
        except Exception as e:
            st.error(f"EDA failed: {e}")

# ------------------------- Feature Importance -------------------------
with tab_importance:
    st.subheader("Model Feature Importance (RandomForest)")
    try:
        model, _, _, _, _, _, ohe_feature_names = load_artifacts()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            k = min(20, len(importances))
            idxs = np.argsort(importances)[::-1][:k]
            top_names = [ohe_feature_names[i] if i < len(ohe_feature_names) else f"feat_{i}" for i in idxs]
            top_vals  = importances[idxs]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(range(k), top_vals[::-1])
            ax.set_yticks(range(k))
            ax.set_yticklabels(top_names[::-1])
            ax.set_xlabel("Importance")
            ax.set_title("Top Feature Importances")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Model does not expose `feature_importances_`.")
    except Exception as e:
        st.error(f"Could not compute importances: {e}")

# ------------------------- Footer -------------------------
st.sidebar.caption("💡 Tip: On Render, keep your `model/` folder in the repo or attach a Persistent Disk for runtime saves.")
