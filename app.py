"""Streamlit app for the Spam / Not Spam email classifier.

Run locally:   streamlit run app.py
Needs spam___model.pkl in the same folder as this file.
"""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "spam___model.pkl"

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def top_words(model, text, n=5):
    """Words in the email that push the prediction most toward spam."""
    vec, clf = model[0], model[-1]
    row = vec.transform([text])
    if row.nnz == 0:
        return []
    names = vec.get_feature_names_out()
    spam_idx = list(clf.classes_).index("spam")
    sign = 1 if spam_idx == 1 else -1
    contrib = row.multiply(clf.coef_[0] * sign).tocoo()
    items = sorted(zip(contrib.col, contrib.data), key=lambda t: t[1], reverse=True)
    return [(names[i], v) for i, v in items[:n] if v > 0]


st.title("📧 Spam Email Detector")
st.caption("TF-IDF + Logistic Regression model. Paste an email below to check it.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Put it next to app.py.")
    st.stop()

model = load_model()

EXAMPLES = {
    "— choose an example —": "",
    "Spam: prize": "Congratulations! You have won a free phone. Click here to claim your prize now.",
    "Spam: account": "URGENT: Your account needs verification. Click the link immediately.",
    "Not spam: meeting": "Hi, can we move our meeting to tomorrow afternoon?",
    "Not spam: leave request": (
        "Subject: Request for Leave\n\nDear Sir/Madam,\n\nI hope you are doing well. "
        "I am writing to request leave for tomorrow due to some personal work.\n\n"
        "Thank you for your understanding.\n\nBest regards"
    ),
}
choice = st.selectbox("Try an example", list(EXAMPLES))
if choice != "— choose an example —":
    st.session_state["email_text"] = EXAMPLES[choice]

email = st.text_area("Email text", key="email_text", height=200,
                     placeholder="Paste the email subject and body here...")

if st.button("Check email", type="primary"):
    if not email.strip():
        st.warning("Please enter some email text first.")
    else:
        pred = model.predict([email])[0]
        proba = dict(zip(model.classes_, model.predict_proba([email])[0]))
        if pred == "spam":
            st.error(f"🚫 Spam — {proba['spam']:.1%} confidence")
        else:
            st.success(f"✅ Not spam — {proba['not spam']:.1%} confidence")

        c1, c2 = st.columns(2)
        c1.metric("Spam probability", f"{proba['spam']:.1%}")
        c2.metric("Not-spam probability", f"{proba['not spam']:.1%}")
        st.progress(float(proba["spam"]))

        words = top_words(model, email)
        if words:
            st.markdown("**Words pushing toward spam:** " + ", ".join(f"`{w}`" for w, _ in words))
        elif pred == "spam" or proba["spam"] > 0.4:
            st.info("The model recognised few words from this email, so treat the result with caution.")

with st.expander("About this model"):
    st.write(
        f"Vocabulary size: {len(model[0].vocabulary_)} words. The model only knows words "
        "seen during training, so unfamiliar emails may be misclassified."
    )
