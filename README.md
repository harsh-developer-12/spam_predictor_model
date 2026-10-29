# 📧 Email Spam Detector

A simple web app that checks whether an email message is **Spam** or **Not Spam**, using a trained machine learning model. Built with Python, Streamlit, scikit-learn, and Joblib.

## How it works

The saved model (`spam_model.pkl`) is a scikit-learn pipeline with two steps:

1. **TF-IDF Vectorizer** converts the email text into numbers.
2. **Logistic Regression** classifies the email as `spam` or `not spam`.

The app loads the pipeline, takes the email text you paste in, and shows the prediction along with the spam probability.

## Project structure

```
project-folder/
├── app.py              # Streamlit app
├── spam_model.pkl      # Trained model (pipeline saved with Joblib)
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

1. Make sure Python 3.9+ is installed.
2. Put `app.py`, `requirements.txt` and `spam_model.pkl` in the same folder.
3. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run the app

From inside the project folder:

```
streamlit run app.py
```

Then open the link shown in the terminal (usually http://localhost:8501), paste an email into the text box and click **Check**.

## Troubleshooting

**"Model file not found"**
- Make sure the model file is named exactly `spam_model.pkl` and is in the same folder as `app.py`.
- On Windows, file extensions may be hidden, so the file can accidentally become `spam_model.pkl.pkl`. Turn on "File name extensions" in File Explorer to check.
- Run `streamlit run app.py` from the project folder.
- Stop the old Streamlit server with `Ctrl + C` and start it again after changing files.

**Version warning or error while loading the model**
- The model was saved with scikit-learn 1.6.1. Install the same version:
  ```
  pip install scikit-learn==1.6.1
  ```

## Notes

- The model's vocabulary is small (around 200 words), so results can be weak on very long or unusual emails.
- To improve it, retrain on a larger dataset and save the new pipeline as `spam_model.pkl`.

## Tech stack

Python, Streamlit, scikit-learn, Joblib, NumPy
