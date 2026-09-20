# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a calibrated SGDClassifier.

🔗 **Live Demo:** [your-app-url.streamlit.app](https://your-app-url.streamlit.app)

---

## 📊 Problem

Credit card fraud is extremely rare but costly:

- 284,807 total transactions
- Only 492 are frauds (**0.172%**)
- A model predicting "never fraud" gets 99.83% accuracy — but catches **zero frauds**

This project builds a model that actually catches fraud while keeping false alarms low.

## 🧠 Approach

1. **EDA** — explored data, confirmed extreme imbalance
2. **Preprocessing** — RobustScaler on `Amount` and `Time`, stratified 80/20 split
3. **Model** — SGDClassifier (`log_loss`, `class_weight='balanced'`)
4. **Calibration** — CalibratedClassifierCV for meaningful probabilities
5. **Threshold tuning** — chose threshold that maximizes F1-score

## 📈 Results

| Metric | Score |
|--------|-------|
| AUPRC | **0.7196** |
| Precision | 0.79 |
| Recall | 0.8061 |
| F1 Score | 0.798 |
| Threshold | 0.1334 |

### Confusion Matrix (Test Set)

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 56,843 | 21 |
| **Actual Fraud** | 19 | **79** |

- ✅ 79 frauds caught (80.6%)
- ❌ 19 frauds missed
- ⚠️ 21 false alarms

### Why AUPRC?

Random guessing gives AUPRC = **0.0017** (the fraud rate). Our model achieves **0.7196** — about **400× better**.

Accuracy is misleading here — a "never fraud" model scores 99.83% accuracy but is useless.

## 🛠️ Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# Setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install
pip install -r requirements.txt

# Run app
streamlit run app.py