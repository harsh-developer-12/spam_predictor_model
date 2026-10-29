from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
DATA_PATH = BASE_DIR / "email_spam_binary_classification_800_rows.csv"

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
)

st.title("📧 Email Spam Detector")
st.write(
    "Enter an email to classify it, or open the Evaluation Dashboard "
    "to review performance on held-out test data."
)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_dataset(path: str):
    df = pd.read_csv(path)
    required = {"subject", "email_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    # Fill missing text fields before combining subject and body.
    df["subject"] = df["subject"].fillna("").astype(str)
    df["email_text"] = df["email_text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # The expected labels are 'spam' and 'not spam'.
    df = df[df["label"].isin(["spam", "not spam"])].copy()
    df["combined_text"] = (
        df["subject"].str.strip() + " " + df["email_text"].str.strip()
    ).str.strip()
    df = df[df["combined_text"].str.len() > 0]

    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both 'spam' and 'not spam' labels.")

    return df


if not MODEL_PATH.exists():
    st.error("Model file not found. Put `spam_model.pkl` beside `app.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))

if not DATA_PATH.exists():
    st.warning(
        "Evaluation dataset not found. Add "
        "`email_spam_binary_classification_800_rows.csv` beside `app.py` "
        "to enable the Evaluation Dashboard. Prediction can still be used."
    )
    dataset = None
else:
    try:
        dataset = load_dataset(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Could not read the evaluation dataset: {exc}")
        dataset = None

prediction_tab, evaluation_tab = st.tabs(
    ["✉️ Predict Email", "📊 Evaluation Dashboard"]
)

with prediction_tab:
    st.subheader("Check an email")
    subject = st.text_input("Email subject (optional)", placeholder="e.g. You won a prize!")
    email_text = st.text_area(
        "Email message",
        height=180,
        placeholder="Paste the email body here...",
    )

    if st.button("Check Email", type="primary"):
        combined_text = f"{subject.strip()} {email_text.strip()}".strip()

        if not combined_text:
            st.warning("Please enter an email subject or message first.")
        else:
            try:
                prediction = str(model.predict([combined_text])[0]).strip().lower()

                if prediction == "spam":
                    st.error("🚨 This email is classified as **Spam**.")
                elif prediction == "not spam":
                    st.success("✅ This email is classified as **Not Spam**.")
                else:
                    st.info(f"Model prediction: **{prediction}**")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([combined_text])[0]
                    classes = [str(label).strip().lower() for label in model.classes_]
                    if "spam" in classes:
                        spam_probability = float(probabilities[classes.index("spam")])
                        st.metric("Spam probability", f"{spam_probability * 100:.1f}%")
                        st.progress(spam_probability)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with evaluation_tab:
    st.subheader("Model Evaluation Dashboard")

    if dataset is None:
        st.info("Add the evaluation CSV beside `app.py` to show the charts.")
    else:
        st.caption(
            "Metrics below use a stratified 80/20 holdout split. The saved model's "
            "pipeline structure is cloned and fitted on the 80% training portion, "
            "then evaluated on the separate 20% test portion. This avoids reporting "
            "training-set scores as test performance."
        )

        X = dataset["combined_text"]
        y = dataset["label"]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

            # Reuse the saved pipeline's estimator configuration, but fit it only
            # on the training portion for a fair holdout evaluation.
            evaluation_model = clone(model)
            evaluation_model.fit(X_train, y_train)
            y_pred = evaluation_model.predict(X_test)
            y_pred = pd.Series(y_pred).astype(str).str.strip().str.lower().to_numpy()

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, pos_label="spam", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, pos_label="spam", zero_division=0
            )
            f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

            st.caption(
                f"Dataset rows used: {len(dataset)} · "
                f"Training rows: {len(X_train)} · Test rows: {len(X_test)}"
            )

            metric_cols = st.columns(4)
            metric_cols[0].metric("Accuracy", f"{accuracy * 100:.2f}%")
            metric_cols[1].metric("Precision (Spam)", f"{precision * 100:.2f}%")
            metric_cols[2].metric("Recall (Spam)", f"{recall * 100:.2f}%")
            metric_cols[3].metric("F1-score (Spam)", f"{f1 * 100:.2f}%")

            st.markdown("### Metrics comparison")
            metrics_df = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
                    "Score": [accuracy, precision, recall, f1],
                }
            ).set_index("Metric")
            st.bar_chart(metrics_df, y="Score", y_label="Score (0 to 1)")

            st.markdown("### Confusion Matrix")
            labels = ["not spam", "spam"]
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(5.5, 4))
            image = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion Matrix")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([0, 1], labels=["Predicted Not Spam", "Predicted Spam"])
            ax.set_yticks([0, 1], labels=["Actual Not Spam", "Actual Spam"])
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

            threshold = cm.max() / 2 if cm.size else 0
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    ax.text(
                        col,
                        row,
                        f"{cm[row, col]}",
                        ha="center",
                        va="center",
                        color="white" if cm[row, col] > threshold else "black",
                        fontsize=12,
                    )

            ax.set_ylabel("Actual label")
            ax.set_xlabel("Predicted label")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()
            st.markdown("### TP, TN, FP and FN")
            detail_cols = st.columns(4)
            detail_cols[0].metric("TN", int(tn))
            detail_cols[1].metric("FP", int(fp))
            detail_cols[2].metric("FN", int(fn))
            detail_cols[3].metric("TP", int(tp))

            st.caption(
                "Note: these are holdout evaluation results for a model refitted on "
                "the training split, using the saved pipeline's configuration. "
                "They are not the training scores of the full-data saved model."
            )
        except Exception as exc:
            st.error(
                "Evaluation could not run. Check that `spam_model.pkl` is a "
                "scikit-learn Pipeline trained on text input and that its expected "
                "input format matches `subject + email_text`."
            )
            st.exception(exc)

st.caption("Built with Python, Streamlit, scikit-learn, pandas, Matplotlib, and Joblib.")
