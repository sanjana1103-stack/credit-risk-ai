import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib/pyplot import

import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from src.preprocessing import preprocess
from src.explain import get_shap_values, plot_waterfall, plot_global_summary, plot_dependence

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Credit Risk AI", page_icon="💰", layout="wide")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #141e30, #243b55); color: white; }
h1, h2, h3 { color: #ffffff; text-align: center; }
label { color: #ffffff !important; }
.stButton>button {
    background-color: #00c6ff; color: black;
    border-radius: 10px; height: 3em; width: 100%;
    font-size: 16px; font-weight: bold;
}
.stButton>button:hover { background-color: #0072ff; color: white; }
.metric-card {
    background: rgba(255,255,255,0.07); border-radius: 12px;
    padding: 16px; text-align: center; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN — credentials from .streamlit/secrets.toml (no hardcoded passwords)
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == st.secrets["auth"]["username"] and password == st.secrets["auth"]["password"]:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS — models load once, not on every Streamlit rerun
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("models/lr_model.pkl"),
        "Random Forest":       joblib.load("models/rf_model.pkl"),
        "XGBoost":             joblib.load("models/xgb_model.pkl"),
    }

@st.cache_data
def load_metrics():
    try:
        with open("models/metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_background_sample():
    """200-row sample used for global SHAP plots."""
    try:
        df = pd.read_csv("data/Loan_default.csv")
        df = preprocess(df)
        return df.drop("Default", axis=1).sample(200, random_state=42)
    except Exception:
        return None

all_models  = load_models()
metrics     = load_metrics()
X_bg_raw    = load_background_sample()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER + SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1>💰 Credit Risk AI System</h1>
<p style='text-align:center;font-size:18px;color:#ccc;'>
Predict loan default risk with Explainable AI
</p>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

if model_choice in ["XGBoost", "Random Forest"]:
    st.sidebar.success("✅ Full SHAP support")
else:
    st.sidebar.warning("⚠️ Limited SHAP support for Logistic Regression")

st.sidebar.markdown("---")
st.sidebar.info("Predicts loan default risk and explains decisions using SHAP.")

model = all_models[model_choice]

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📊 Model Metrics",
    "🌍 Global SHAP",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📝 Applicant Information")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age          = st.number_input("Age", 18, 100, value=30)
        income       = st.number_input("Income", min_value=0.0, value=50000.0)
        loan_amount  = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        education    = st.selectbox("Education", ["High School", "Bachelor's", "Master's"])
        employment   = st.selectbox("Employment Type", ["Unemployed", "Part-time", "Full-time"])

    with col2:
        months_employed  = st.number_input("Months Employed", min_value=0, value=24)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
        interest_rate    = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
        loan_term        = st.number_input("Loan Term (months)", min_value=1, value=36)
        dti_ratio        = st.number_input("DTI Ratio (0–1)", min_value=0.0, max_value=1.0, value=0.3)
        marital          = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        mortgage         = st.selectbox("Has Mortgage", ["No", "Yes"])
        dependents       = st.selectbox("Has Dependents", ["No", "Yes"])
        purpose          = st.selectbox("Loan Purpose", ["Auto", "Business", "Home", "Other"])
        cosigner         = st.selectbox("Has Co-Signer", ["No", "Yes"])

    if st.button("🔍 Predict Risk"):
        input_df = pd.DataFrame([{
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "CreditScore": credit_score, "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
            "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": education, "EmploymentType": employment,
            "MaritalStatus": marital, "HasMortgage": mortgage,
            "HasDependents": dependents, "LoanPurpose": purpose,
            "HasCoSigner": cosigner,
        }])

        input_df = preprocess(input_df)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_df)[0]
        prob       = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.markdown("<h2 style='color:#ff4b4b;'>⚠️ High Risk</h2>", unsafe_allow_html=True)
            st.error("This applicant is likely to default.")
        else:
            st.markdown("<h2 style='color:#00ffcc;'>✅ Low Risk</h2>", unsafe_allow_html=True)
            st.success("This applicant is financially safe.")

        st.markdown("### 📊 Risk Score")
        st.progress(float(prob))
        st.markdown(
            f"<h3 style='text-align:center;color:#00c6ff;'>Default Probability: {prob:.2%}</h3>",
            unsafe_allow_html=True
        )

        # SHAP explanation — only for tree models; scoped so shap_vals never leaks
        if model_choice in ["XGBoost", "Random Forest"]:
            st.markdown("## 🔍 AI Explanation (Waterfall)")
            shap_values        = get_shap_values(model, input_df)
            fig_wf, shap_vals  = plot_waterfall(shap_values, model_choice)
            st.pyplot(fig_wf)
            plt.clf()

            impact_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Impact":  shap_vals,
            }).sort_values("Impact", ascending=False)

            st.markdown("## 🧠 Explanation Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.write("#### 🔺 Top Risk Drivers")
                for _, row in impact_df.head(3).iterrows():
                    st.write(f"- **{row['Feature']}** (`{row['Impact']:+.3f}`)")
            with c2:
                st.write("#### 🔻 Top Risk Reducers")
                for _, row in impact_df.tail(3).iterrows():
                    st.write(f"- **{row['Feature']}** (`{row['Impact']:+.3f}`)")

            # Feature importance — scoped inside the tree-model block (bug fix)
            st.markdown("## 📊 Feature Importance (this prediction)")
            imp_df = (
                pd.DataFrame({"Feature": input_df.columns, "Importance": abs(shap_vals)})
                .sort_values("Importance", ascending=False)
                .head(10)
            )
            st.bar_chart(imp_df.set_index("Feature"))

        else:
            st.info("SHAP waterfall is not reliable for Logistic Regression. "
                    "Switch to XGBoost or Random Forest for full explainability.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📂 Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV with the same columns as the single prediction form.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw_df)} applicants loaded.** Preview:")
        st.dataframe(raw_df.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Predictions"):
            try:
                proc_df = preprocess(raw_df.copy())
                proc_df = proc_df.reindex(columns=model.feature_names_in_, fill_value=0)

                preds = model.predict(proc_df)
                probs = model.predict_proba(proc_df)[:, 1]

                result_df = raw_df.copy()
                result_df["Default_Probability"] = probs.round(4)
                result_df["Risk_Label"] = (
                    pd.Series(preds)
                    .map({0: "✅ Low Risk", 1: "⚠️ High Risk"})
                    .values
                )

                st.markdown("### 📋 Results")
                st.dataframe(
                    result_df[["Default_Probability", "Risk_Label"] + list(raw_df.columns)],
                    use_container_width=True,
                )

                n_high = int(preds.sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Applicants", len(preds))
                c2.metric("⚠️ High Risk",      n_high)
                c3.metric("✅ Low Risk",       len(preds) - n_high)

                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during batch prediction: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · MODEL METRICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Model Performance Dashboard")

    if not metrics:
        st.warning("No metrics found. Run `src/train.py` first to generate `models/metrics.json`.")
    else:
        cols = st.columns(len(metrics))
        for col, (name, m) in zip(cols, metrics.items()):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>{name}</h4>
                    <p>AUC-ROC &nbsp;<b>{m['test_auc']}</b></p>
                    <p>Avg Precision &nbsp;<b>{m['avg_precision']}</b></p>
                    <p style='font-size:13px;color:#aaa;'>
                        CV AUC: {m['cv_auc_mean']} ± {m['cv_auc_std']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        metric_df = (
            pd.DataFrame(metrics)
            .T.reset_index()
            .rename(columns={"index": "Model"})
        )
        st.markdown("#### AUC-ROC vs Average Precision")
        st.bar_chart(metric_df.set_index("Model")[["test_auc", "avg_precision"]])
        st.markdown("#### Full Metrics Table")
        st.dataframe(metric_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · GLOBAL SHAP
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🌍 Global SHAP Analysis")
    st.markdown("Computed across a **200-row background sample** from the training data.")

    if model_choice not in ["XGBoost", "Random Forest"]:
        st.warning("Global SHAP requires XGBoost or Random Forest. Switch the model in the sidebar.")
    elif X_bg_raw is None:
        st.warning("Training data not found at `data/Loan_default.csv`. "
                   "Place the dataset there for global SHAP plots.")
    else:
        X_bg = X_bg_raw.reindex(columns=model.feature_names_in_, fill_value=0)

        if st.button("📊 Generate Global SHAP Summary"):
            with st.spinner("Computing SHAP values…"):
                fig_gs = plot_global_summary(model, X_bg, model_choice)
                st.pyplot(fig_gs)
                plt.clf()

        st.markdown("---")
        st.markdown("#### 🔗 SHAP Dependence Plot")
        dep_feature = st.selectbox("Select feature", options=list(model.feature_names_in_))

        if st.button("📈 Generate Dependence Plot"):
            with st.spinner(f"Computing dependence plot for {dep_feature}…"):
                fig_dp = plot_dependence(model, X_bg, dep_feature, model_choice)
                st.pyplot(fig_dp)
                plt.clf()
