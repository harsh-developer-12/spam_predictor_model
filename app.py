from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("📧 Spam Email Detector")
st.write("Email ka text paste karo aur check karo ki wo spam hai ya nahi.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found. Put `spam_model.pkl` beside `app.py` ({BASE_DIR}).")
    st.stop()

model = load_model()

email_text = st.text_area("Email text", height=220, placeholder="Yahan email likho ya paste karo...")

if st.button("Check", type="primary"):
    if not email_text.strip():
        st.warning("Pehle email text daalo.")
    else:
        prediction = model.predict([email_text])[0]
        proba = model.predict_proba([email_text])[0]
        confidence = proba[list(model.classes_).index(prediction)] * 100

        if prediction == "spam":
            st.error(f"🚨 Spam Email  (confidence: {confidence:.1f}%)")
        else:
            st.success(f"✅ Not Spam  (confidence: {confidence:.1f}%)")
