# 📧 Email Spam Classification Using Machine Learning

A beginner-friendly Machine Learning project that classifies email text as **Spam** or **Not Spam**. This project was developed and tested in a **Kaggle Notebook**.

## 👨‍💻 Author

**Harsh Pandey**

- GitHub Profile: [harsh-developer-12](https://github.com/harsh-developer-12)
- Project Repository: [spam_predictor_model](https://github.com/harsh-developer-12/spam_predictor_model)

## 📌 Project Overview

Email spam classification is a binary classification problem. The model takes email text as input and predicts one of two labels:

- **Spam** — an email classified as unwanted or suspicious.
- **Not Spam** — an email classified as legitimate.

This project covers data preparation, text feature extraction, model training, evaluation, saving the trained model, and predicting new email text.

## ✨ Features

- Load and clean email data with Pandas
- Handle missing values and duplicate records
- Combine email subject and email body text
- Convert text into numeric features using TF-IDF
- Train a text classification model
- Evaluate performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Save and load the trained model with Joblib
- Enter new email text and predict its class

## 🧰 Tools & Libraries

- Python
- Kaggle Notebooks
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Joblib
- Matplotlib (for visualizations, if used in the notebook)

## 🔄 Workflow

1. Load the dataset.
2. Inspect and clean the data.
3. Handle missing values and remove duplicate records.
4. Prepare the text features and target labels.
5. Split the data into training and testing sets.
6. Convert email text into numerical features using TF-IDF.
7. Train the classification model.
8. Evaluate the model using classification metrics.
9. Save the trained model to a `.pkl` file using Joblib.
10. Load the saved model and predict the class of new email text.

## 🧪 Example: Load the Saved Model and Predict

The following example assumes the saved file is named `spam_model.pkl` and contains the complete text-processing pipeline and classifier.

```python
import joblib

# Load the saved model
loaded_model = joblib.load("spam_model.pkl")

# Enter an email to classify
new_email = input("Enter your email: ")

# Predict the email label
prediction = loaded_model.predict([new_email])

if prediction[0] == "spam":
    print("Result: Spam Email")
else:
    print("Result: Not Spam Email")
```

> **Note:** This example assumes the model's output labels are exactly `spam` and `not spam` (lowercase). Update the label check if your dataset uses different label names. The saved model should include the same TF-IDF preprocessing used during training, typically as a Scikit-learn `Pipeline`.

## 📊 Evaluation Metrics

| Metric | What it measures |
| --- | --- |
| Accuracy | Proportion of all predictions that are correct |
| Precision | Of emails predicted as spam, how many were actually spam |
| Recall | Of actual spam emails, how many were detected |
| F1-score | Harmonic mean of Precision and Recall |
| ROC-AUC | How well the model distinguishes between the two classes using prediction scores/probabilities |

Actual metric values depend on the dataset, train/test split, and trained model. They are not listed here because they have not been verified for this README.

## ▶️ Run the Project

1. Open the notebook in Kaggle.
2. Attach or upload the dataset required by the notebook.
3. Run the notebook cells in order.
4. Review the evaluation metrics.
5. Save the trained pipeline with Joblib before loading it for new predictions.

If you run the project locally, install the libraries used in your notebook. For example:

```bash
pip install pandas scikit-learn joblib matplotlib
```

## 📁 Suggested Repository Structure

Adjust these names to match the files actually uploaded to your repository.

```text
spam_predictor_model/
├── spam_classifier.ipynb
├── spam_model.pkl
├── email_spam_binary_classification_800_rows.csv
└── README.md
```

## 🚀 Future Improvements

- Build an interactive web interface using Streamlit
- Compare multiple classification algorithms on the same train/test split
- Improve text preprocessing and evaluate misclassified emails
- Deploy the application online

## ⚠️ Limitations

This model is a learning project and may misclassify emails. Do not rely on its prediction as the only way to determine whether an email is safe or legitimate. Avoid opening suspicious links or attachments.

## 📚 Learning Outcomes

This project provided practice with data cleaning, text preprocessing, TF-IDF feature extraction, binary classification, evaluation metrics, and saving/loading a trained Machine Learning model.

---

⭐ Thanks for visiting my project!
