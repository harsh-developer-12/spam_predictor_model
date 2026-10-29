from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

st.title("📧 Email Spam Detector")
st.write(
    "Enter an email message below to check whether your trained model "
    "classifies it as Spam or Not Spam."
)


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put `spam_model.pkl` in the same folder as "
        "`app.py`, then restart the app."
    )
    st.stop()

model = load_model(MODEL_PATH)

email_text = st.text_area("Email message", height=200, placeholder="Paste the email text here...")

if st.button("Check"):
    if not email_text.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([email_text])[0]
        proba = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        spam_prob = proba[classes.index("spam")] if "spam" in classes else None

        if str(prediction).lower() == "spam":
            st.error("🚨 This email is classified as **Spam**.")
        else:
            st.success("✅ This email is classified as **Not Spam**.")

        if spam_prob is not None:
            st.metric("Spam probability", f"{spam_prob * 100:.1f}%")
            st.progress(float(spam_prob))

st.caption("Built with Python, Streamlit, scikit-learn, and Joblib.")
