import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(
    page_title="Bankruptcy Prediction Tool",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏦 Bankruptcy Prediction Tool")

# =========================================================
# MODEL SELECTION
# =========================================================

st.sidebar.header("🧠 Model Selection")

MODEL_DIR = "models"

# Find available models
if not os.path.exists(MODEL_DIR):
    st.error(f"Model directory '{MODEL_DIR}' not found.")
    st.stop()

model_files = [
    f for f in os.listdir(MODEL_DIR)
    if f.endswith(".pkl") or f.endswith(".joblib")
]

if not model_files:
    st.error("No trained models found in the 'models/' folder.")
    st.stop()

selected_model_file = st.sidebar.selectbox(
    "Select a trained model",
    model_files
)

MODEL_PATH = os.path.join(MODEL_DIR, selected_model_file)

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
    st.sidebar.success(f"Loaded model: {selected_model_file}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# =========================================================
# FEATURE NAMES
# =========================================================

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    st.error("Model does not contain feature names.")
    st.stop()

st.write(f"**Number of required features:** {len(feature_names)}")

with st.expander("🔍 View required features"):
    st.write(feature_names)

# =========================================================
# FILE UPLOAD
# =========================================================

st.header("📂 Upload CSV File")

uploaded_file = st.file_uploader(
    "Upload a CSV file with financial features",
    type=["csv"]
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_from_csv(data, threshold=0.10):
    X = data[feature_names]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    return pd.DataFrame({
        "Company_ID": range(1, len(data) + 1),
        "Prediction": np.where(preds == 1, "Bankrupt", "Not Bankrupt"),
        "Bankruptcy_Probability (%)": (probs * 100).round(2),
        "Risk_Level": np.where(
            probs >= threshold * 3, "High",
            np.where(probs >= threshold * 1.5, "Medium", "Low")
        ),
        "Confidence (%)": (np.abs(probs - 0.5) * 200).round(2)
    })

# =========================================================
# MAIN LOGIC
# =========================================================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Preview of uploaded data")
    st.dataframe(df.head())

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        st.error(f"Missing {len(missing)} required features.")
        st.write(missing)
        st.stop()

    st.success("All required features found")

    # Sidebar threshold
    st.sidebar.header("⚙️ Prediction Settings")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01
    )

    if st.button("🚀 Make Predictions"):
        results = predict_from_csv(df, threshold)

        # =================================================
        # RESULTS
        # =================================================

        st.header("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Companies", len(results))
        col2.metric("Bankrupt", (results["Prediction"] == "Bankrupt").sum())
        col3.metric("Not Bankrupt", (results["Prediction"] == "Not Bankrupt").sum())

        st.dataframe(results)

        # =================================================
        # VISUALIZATIONS
        # =================================================

        st.header("📈 Visualizations")

        col1, col2 = st.columns(2)

        # Prediction distribution
        with col1:
            fig, ax = plt.subplots()
            pred_counts = results["Prediction"].value_counts()
            ax.bar(pred_counts.index, pred_counts.values, color=["red", "green"])
            ax.set_title("Prediction Distribution")
            ax.set_ylabel("Count")
            ax.set_xticks(range(len(pred_counts.index)))
            ax.set_xticklabels(pred_counts.index, rotation=0)
            st.pyplot(fig)

        # Risk level distribution
        with col2:
            fig, ax = plt.subplots()
            risk_counts = results["Risk_Level"].value_counts().reindex(
                ["Low", "Medium", "High"]
            )
            ax.bar(risk_counts.index, risk_counts.values, color=["green", "orange", "red"])
            ax.set_title("Risk Level Distribution")
            ax.set_ylabel("Count")
            ax.set_xticks(range(len(risk_counts.index)))
            ax.set_xticklabels(risk_counts.index, rotation=0)
            st.pyplot(fig)

        # Probability histogram
        fig, ax = plt.subplots()
        ax.hist(results["Bankruptcy_Probability (%)"], bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(threshold * 100, color="red", linestyle="--", label="Threshold")
        ax.set_title("Bankruptcy Probability Distribution")
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        # =================================================
        # DOWNLOAD
        # =================================================

        st.header("⬇️ Download Results")
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions as CSV",
            csv,
            file_name="bankruptcy_predictions.csv",
            mime="text/csv"
        )
