import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# ---------- Page config ----------
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# ---------- Load model ----------
@st.cache_resource
def load_bundle():
    return joblib.load('models/fraud_model_bundle.pkl')

bundle = load_bundle()
model = bundle['model']
threshold = bundle['threshold']
metrics = bundle['metrics']

# ---------- Header ----------
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
    This app uses an **SGDClassifier (calibrated)** trained on the 
    [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
    It predicts whether a transaction is **fraudulent** or **normal**.
    """
)

# ---------- Metrics row ----------
st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUPRC", metrics['AUPRC'])
col2.metric("Precision", metrics['Precision'])
col3.metric("Recall", metrics['Recall'])
col4.metric("F1 Score", metrics['F1'])

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📈 Model Insights", "ℹ️ About"])

# ============ TAB 1: PREDICT ============
with tab1:
    st.subheader("Enter Transaction Details")
    
    col_a, col_b = st.columns(2)
    with col_a:
        amount = st.number_input("Transaction Amount (€)", min_value=0.0, value=100.0, step=1.0)
    with col_b:
        time_val = st.number_input("Time (seconds since first txn)", min_value=0.0, value=50000.0, step=100.0)

    st.markdown("### PCA Features (V1–V28)")
    st.caption("These are anonymized PCA-transformed features from the original dataset.")
    
    # Use a compact input method
    with st.expander("Enter V1–V28 values (optional)"):
        v_cols = st.columns(4)
        v_values = []
        for i in range(28):
            with v_cols[i % 4]:
                v = st.number_input(f"V{i+1}", value=0.0, step=0.1, key=f"v{i+1}")
                v_values.append(v)
    
    if st.button("🚨 Check Transaction", type="primary"):
        # Build the input row
        input_dict = {'scaled_amount': amount, 'scaled_time': time_val}
        for i, v in enumerate(v_values):
            input_dict[f'V{i+1}'] = v
        
        input_df = pd.DataFrame([input_dict])
        
        # Ensure column order matches training
        input_df = input_df[model.feature_names_in_]
        
        # Predict
        proba = model.predict_proba(input_df)[0, 1]
        is_fraud = proba >= threshold
        
        # Display result
        st.markdown("### Result")
        if is_fraud:
            st.error(f"⚠️ **FRAUD DETECTED** — Probability: {proba:.2%}")
        else:
            st.success(f"✅ **Normal Transaction** — Fraud Probability: {proba:.2%}")
        
        st.progress(float(proba))
        st.caption(f"Decision threshold: {threshold:.4f}")

# ============ TAB 2: INSIGHTS ============
with tab2:
    st.subheader("Model Insights")
    
    st.markdown("#### Confusion Matrix (on test set)")
    cm_data = {
        '': ['Actual Normal', 'Actual Fraud'],
        'Predicted Normal': [metrics['True Negatives'], metrics['False Negatives']],
        'Predicted Fraud':  [metrics['False Positives'], metrics['True Positives']]
    }
    st.dataframe(pd.DataFrame(cm_data).set_index(''), use_container_width=True)
    
    st.markdown("#### What these mean")
    st.markdown(f"""
    - **True Positives**: {metrics['True Positives']} frauds caught ✅
    - **False Positives**: {metrics['False Positives']} false alarms (normal flagged as fraud)
    - **False Negatives**: {metrics['False Negatives']} missed frauds ⚠️
    - **True Negatives**: {metrics['True Negatives']} normal transactions correctly ignored
    """)

# ============ TAB 3: ABOUT ============
with tab3:
    st.subheader("About this project")
    st.markdown("""
    ### Dataset
    - 284,807 transactions, 492 frauds (0.172%)
    - Features V1–V28 are PCA-transformed
    - Only `Time` and `Amount` are original
    
    ### Model
    - **Algorithm**: SGDClassifier with `log_loss`
    - **Imbalance handling**: `class_weight='balanced'`
    - **Calibration**: `CalibratedClassifierCV` for meaningful probabilities
    - **Metric**: AUPRC (Area Under Precision-Recall Curve)
    
    ### Why AUPRC?
    Because accuracy is meaningless on imbalanced data — a model predicting "no fraud" 
    always would score 99.83% accuracy but catch zero frauds.
    
    ### Threshold
    Chosen to maximize F1-score on the test set.
    """)