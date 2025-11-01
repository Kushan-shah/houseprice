# app.py — Real Estate House Price Prediction (Streamlit)
# Loads saved Random Forest model and preprocessing artifacts (scaler, encoder) for predictions and EDA.

import os
import io
import pickle
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------- Page Config & Styling ---------------------
st.set_page_config(page_title="Real Estate House Price Prediction", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    .title-container { text-align: center; padding: 10px 0 20px 0; }
    .title-text { font-size: 38px; font-weight: 800; }
    .subtitle-text { font-size: 18px; color: #666; }
    .card { background: #ffffff; padding: 16px; border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --------------------- Header ---------------------
st.markdown("""
<div class="title-container">
    <div class="title-text">🏡 Real Estate House Price Prediction</div>
    <div class="subtitle-text">Predict property values using Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# --------------------- Paths to Artifacts ---------------------
ART_DIR = "model"
MODEL_PATH  = os.path.join(ART_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
ENC_PATH    = os.path.join(ART_DIR, "onehot_encoder.pkl")
TRAIN_COLS  = os.path.join(ART_DIR, "training_columns.pkl")
NUM_COLS    = os.path.join(ART_DIR, "num_cols.pkl")
CAT_COLS    = os.path.join(ART_DIR, "cat_cols.pkl")

# --------------------- Artifact Loader ---------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(ENC_PATH, "rb") as f: encoder = pickle.load(f)
    with open(TRAIN_COLS, "rb") as f: training_columns = pickle.load(f)
    with open(NUM_COLS, "rb") as f: num_cols = pickle.load(f)
    with open(CAT_COLS, "rb") as f: cat_cols = pickle.load(f)

    # Get OneHot feature names
    try:
        cat_new = encoder.get_feature_names_out(cat_cols)
    except:
        cat_new = encoder.get_feature_names(cat_cols)
    feature_names = list(num_cols) + list(cat_new)

    return model, scaler, encoder, training_columns, num_cols, cat_cols, feature_names

# --------------------- Data Cleaning ---------------------
def context_aware_impute(df):
    df = df.copy()
    none_cols = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
                 "GarageType","GarageFinish","GarageQual","GarageCond",
                 "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
                 "MasVnrType"]
    zero_cols = ["GarageYrBlt","GarageArea","GarageCars","BsmtFinSF1","BsmtFinSF2",
                 "BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","MasVnrArea"]

    for col in none_cols:
        if col in df: df[col] = df[col].fillna("None")
    for col in zero_cols:
        if col in df: df[col] = df[col].fillna(0)

    if "LotFrontage" in df and "Neighborhood" in df:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(df[col].median())

    return df

def feature_engineer(df):
    df = df.copy()
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    return df

# --------------------- Final Transform Before Prediction ---------------------
def transform_for_model(df, scaler, encoder, train_cols, num_cols, cat_cols):
    df = df.reindex(columns=train_cols)

    if num_cols:
        num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        X_num = scaler.transform(num)
    else:
        X_num = np.empty((len(df), 0))

    if cat_cols:
        cat = df[cat_cols].astype(str).replace({"nan":"None"}).fillna("None")
        X_cat = encoder.transform(cat)
        if hasattr(X_cat, "toarray"): X_cat = X_cat.toarray()
    else:
        X_cat = np.empty((len(df), 0))

    return np.hstack([X_num, X_cat]) if X_cat.size else X_num

def predict_prices(input_df):
    model, scaler, encoder, train_cols, num_cols, cat_cols, _ = load_artifacts()
    df = context_aware_impute(input_df.copy())
    df = feature_engineer(df)
    X = transform_for_model(df, scaler, encoder, train_cols, num_cols, cat_cols)
    return np.expm1(model.predict(X))  # log->price

# --------------------- Tabs ---------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Predict", "📂 Batch CSV", "📊 EDA", "⭐ Feature Importance"])

# --------------------- Tab 1: Single Prediction ---------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter House Details")

    col1, col2 = st.columns(2)
    with col1:
        GrLivArea = st.number_input("Ground Living Area (sqft)", 500, 7000, 1500, step=50)
        TotalBsmtSF = st.number_input("Basement Area (sqft)", 0, 4000, 800, step=50)
        FirstFlr = st.number_input("1st Floor (sqft)", 0, 4000, 900)
        SecondFlr = st.number_input("2nd Floor (sqft)", 0, 3000, 600)
    with col2:
        FullBath = st.number_input("Full Bathrooms", 0, 5, 2)
        HalfBath = st.number_input("Half Bathrooms", 0, 5, 1)
        OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        GarageCars = st.slider("Garage Capacity (Cars)", 0, 5, 2)

    YrSold = st.number_input("Year Sold", 2006, 2010, 2009)
    YearBuilt = st.number_input("Year Built", 1800, 2025, 2003)
    Neighborhood = st.text_input("Neighborhood", "NAmes")
    ExterQual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"])

    if st.button("Predict House Price", use_container_width=True):
        try:
            df = pd.DataFrame([{
                "GrLivArea": GrLivArea, "TotalBsmtSF": TotalBsmtSF, "1stFlrSF": FirstFlr, "2ndFlrSF": SecondFlr,
                "FullBath": FullBath, "HalfBath": HalfBath, "OverallQual": OverallQual, "GarageCars": GarageCars,
                "YrSold": YrSold, "YearBuilt": YearBuilt, "Neighborhood": Neighborhood, "ExterQual": ExterQual
            }])
            price = predict_prices(df)[0]
            st.success(f"💰 Estimated House Price: **${price:,.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
# --------------------- Tab 2: Batch CSV Prediction ---------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload CSV for Batch Prediction")
    st.markdown("Upload a CSV containing the same feature columns used in training (no `SalePrice`).")

    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            # Keep Id if present; otherwise create a simple sequence
            id_series = df_in["Id"] if "Id" in df_in.columns else pd.Series(np.arange(1, len(df_in)+1), name="Id")
            preds = predict_prices(df_in)
            out = pd.DataFrame({"Id": id_series, "SalePrice": preds})
            st.write("Preview of predictions:", out.head())

            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download predictions.csv",
                data=buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------- Tab 3: Simple EDA ---------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Quick EDA (Upload any CSV)")

    eda_file = st.file_uploader("Upload a CSV for EDA", type=["csv"], key="eda_uploader")
    if eda_file is not None:
        try:
            df_eda = pd.read_csv(eda_file)
            st.write("Data Preview:", df_eda.head())

            # Numeric histograms (up to 9)
            num_df = df_eda.select_dtypes(include=[np.number])
            if num_df.shape[1] > 0:
                st.markdown("**Numeric Distributions**")
                cols = st.columns(3)
                for i, col in enumerate(num_df.columns[:9]):
                    with cols[i % 3]:
                        fig, ax = plt.subplots(figsize=(3.6, 3))
                        ax.hist(num_df[col].dropna().values, bins=30)
                        ax.set_title(col)
                        st.pyplot(fig, clear_figure=True)

                # Correlation heatmap
                if num_df.shape[1] >= 2:
                    st.markdown("**Correlation Heatmap**")
                    corr = num_df.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    im = ax.imshow(corr.values, aspect="auto")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=90)
                    ax.set_yticklabels(corr.columns)
                    fig.colorbar(im, fraction=0.046, pad=0.04)
                    ax.set_title("Correlation Heatmap")
                    st.pyplot(fig, clear_figure=True)
            else:
                st.info("No numeric columns detected.")
        except Exception as e:
            st.error(f"EDA error: {e}")
    else:
        st.info("Upload a CSV to see distributions and correlations.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------- Tab 4: Feature Importance ---------------------
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Feature Importance")

    try:
        model, _, _, _, _, _, feature_names = load_artifacts()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Top-K
            K = min(20, len(importances))
            idx = np.argsort(importances)[::-1][:K]
            names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in idx]
            vals  = importances[idx]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(range(K), vals[::-1])
            ax.set_yticks(range(K))
            ax.set_yticklabels(names[::-1])
            ax.set_xlabel("Importance")
            ax.set_title("Top Feature Importances (Random Forest)")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("This model does not expose `feature_importances_`.")
    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------- Footer ---------------------
st.write("")
st.caption("© Real Estate House Price Prediction — Powered by Machine Learning")
