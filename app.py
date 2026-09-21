import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Page config ----------
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# ---------- Load model bundle ----------
@st.cache_resource
def load_bundle():
    return joblib.load('models/fraud_model_xgb_bundle.pkl')

bundle = load_bundle()
model        = bundle['model']
preprocessor = bundle['preprocessor']
threshold    = bundle['threshold']
metrics      = bundle['metrics']

# ---------- Header ----------
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "Detect fraudulent transactions using an **XGBoost classifier** "
    "trained on the [Sparkov credit card fraud dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)."
)

# ---------- Top metrics ----------
st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUPRC",     metrics['AUPRC'])
c2.metric("Precision", metrics['Precision'])
c3.metric("Recall",    metrics['Recall'])
c4.metric("F1 Score",  metrics['F1'])

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📈 Insights", "ℹ️ About"])

# ============ TAB 1: PREDICT ============
with tab1:
    st.subheader("Enter Transaction Details")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0)
        category = st.selectbox("Category", [
            'grocery_pos', 'gas_transport', 'misc_net', 'grocery_net',
            'shopping_net', 'shopping_pos', 'food_dining', 'personal_care',
            'health_fitness', 'home', 'kids_pets', 'entertainment',
            'travel', 'misc_pos'
        ])
        gender = st.selectbox("Gender", ['M', 'F'])

    with col_b:
        trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
        trans_day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        trans_month = st.slider("Month", 1, 12, 6)
        trans_is_weekend = st.checkbox("Is Weekend?", value=False)

    with col_c:
        age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=40)
        city_pop = st.number_input("City Population", min_value=0, value=50000, step=1000)
        state = st.text_input("State (2-letter code)", value="CA")

    # Advanced numeric features (collapsed)
    with st.expander("Advanced: Location & Merchant Coordinates"):
        ca, cb = st.columns(2)
        with ca:
            lat = st.number_input("Customer Latitude", value=40.0, step=0.01)
            long = st.number_input("Customer Longitude", value=-74.0, step=0.01)
        with cb:
            merch_lat = st.number_input("Merchant Latitude", value=40.0, step=0.01)
            merch_long = st.number_input("Merchant Longitude", value=-74.0, step=0.01)

    if st.button("🚨 Check Transaction", type="primary"):
        # Compute engineered features
        amt_log = np.log1p(amt)
        amt_is_round = int(amt % 1 == 0)
        trans_is_night = int(trans_hour in [0, 1, 2, 3, 4, 5])
        distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)

        # Build input row with all required columns
        input_dict = {
            'amt': amt,
            'city_pop': city_pop,
            'lat': lat,
            'long': long,
            'merch_lat': merch_lat,
            'merch_long': merch_long,
            'trans_hour': trans_hour,
            'trans_day_of_week': trans_day_of_week,
            'trans_month': trans_month,
            'trans_is_weekend': int(trans_is_weekend),
            'trans_is_night': trans_is_night,
            'amt_log': amt_log,
            'amt_is_round': amt_is_round,
            'distance': distance,
            'age': age,
            'category': category,
            'gender': gender,
            'state': state
        }

        input_df = pd.DataFrame([input_dict])

        # Apply the same preprocessing pipeline
        try:
            X_processed = preprocessor.transform(input_df)

            # Predict
            proba = model.predict_proba(X_processed)[0, 1]
            is_fraud = proba >= threshold

            st.markdown("### Result")
            if is_fraud:
                st.error(f"⚠️ **FRAUD DETECTED** — Probability: {proba:.2%}")
            else:
                st.success(f"✅ **Normal Transaction** — Fraud Probability: {proba:.2%}")

            st.progress(float(min(proba, 1.0)))
            st.caption(f"Decision threshold: {threshold:.4f}")

            # Show key features used
            with st.expander("View input features sent to model"):
                st.dataframe(input_df.T.rename(columns={0: 'value'}))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Make sure the column names match what the preprocessor expects.")

# ============ TAB 2: INSIGHTS ============
with tab2:
    st.subheader("Model Insights")

    st.markdown("#### Confusion Matrix (test set)")
    cm_df = pd.DataFrame(
        [[metrics['True Negatives'], metrics['False Positives']],
         [metrics['False Negatives'], metrics['True Positives']]],
        columns=['Predicted Normal', 'Predicted Fraud'],
        index=['Actual Normal', 'Actual Fraud']
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown(f"""
    - **True Positives (caught frauds)**: {metrics['True Positives']} ✅
    - **False Positives (false alarms)**: {metrics['False Positives']}
    - **False Negatives (missed frauds)**: {metrics['False Negatives']} ⚠️
    - **True Negatives (correct normals)**: {metrics['True Negatives']}
    """)

    st.markdown("---")
    st.markdown("### Why XGBoost?")
    st.markdown("""
    XGBoost is a **gradient-boosted tree ensemble** that:
    - Captures **non-linear interactions** between features (e.g., high amount + late night + certain category)
    - Handles **high-cardinality categorical features** well
    - Is **robust to outliers** in transaction amounts
    - Consistently outperforms linear models on tabular fraud data
    """)

# ============ TAB 3: ABOUT ============
with tab3:
    st.subheader("About this project")
    st.markdown(f"""
    ### Dataset
    - **1.3M** transactions from the Sparkov simulation
    - Fraud rate: **~0.58%**
    - Readable features: `amt`, `category`, `merchant`, `city_pop`, `job`, etc.

    ### Pipeline
    1. **Feature engineering**: extracted hour, day, month, age, distance, log-amount
    2. **Preprocessing**: RobustScaler on numerics + OneHotEncoder on categoricals
    3. **Model**: XGBoost with `scale_pos_weight` for class imbalance
    4. **Threshold tuning**: chosen to maximize F1 on the PR curve

    ### Current model
    - **AUPRC**: {metrics['AUPRC']}
    - **Precision**: {metrics['Precision']}
    - **Recall**: {metrics['Recall']}
    - **F1**: {metrics['F1']}
    - **Threshold**: {threshold:.4f}

    ### Why AUPRC?
    Accuracy is misleading on imbalanced data. A model predicting "no fraud" always
    would score 99.42% accuracy but catch **zero frauds**.

    ### Why readable features matter
    Unlike the PCA-anonymized dataset (`V1-V28`), this dataset has **interpretable
    features** — we can explain *why* a transaction was flagged.
    """)