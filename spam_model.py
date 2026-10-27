import streamlit as st
import joblib
from pathlib import Path

# -----------------------------
# Email Spam Classification App
# -----------------------------

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"

@st.cache_resource
def load_model():
    """Load the saved text-classification pipeline."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.title("📧 Email Spam Detector")
st.write(
    "Enter an email message below to check whether your trained model "
    "classifies it as Spam or Not Spam."
)

model = load_model()

if model is None:
    st.error(
        "Model file not found. Put `spam_model.pkl` in the same folder "
        "as `app.py`, then restart the app."
    )
else:
    email_text = st.text_area(
        "Email text",
        placeholder="Paste or type the email message here...",
        height=180
    )

    if st.button("Check Email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please enter an email message first.")
        else:
            try:
                prediction = model.predict([email_text])[0]
                label = str(prediction).strip().lower()

                if label == "spam":
                    st.error("🚨 Prediction: Spam Email")
                elif label in {"not spam", "not_spam", "ham", "not-spam"}:
                    st.success("✅ Prediction: Not Spam Email")
                else:
                    st.info(f"Prediction: {prediction}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([email_text])[0]
                    classes = list(model.classes_)
                    st.caption("Model class probabilities")
                    for class_name, probability in zip(classes, probabilities):
                        st.write(f"{class_name}: {probability:.1%}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "Check that `spam_model.pkl` contains the complete fitted "
                    "text-processing pipeline (for example, TF-IDF + classifier), "
                    "and that its expected input matches the email text."
                )

st.divider()
st.caption("Built with Python, Streamlit, scikit-learn, and Joblib.")
