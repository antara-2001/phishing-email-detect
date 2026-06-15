import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="📧",
    layout="wide"
)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------

@st.cache_resource
def load_model():
    model = joblib.load("phishing_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_model()

# -----------------------------
# LOAD METRICS
# -----------------------------

try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
except:
    metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "roc_auc": 0
    }

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# HEADER
# -----------------------------

st.title("AI-Powered Phishing Email Detection")

st.markdown("""
Detect whether an email is **Phishing** or **Legitimate**
using Natural Language Processing and Machine Learning.
""")

st.divider()

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.subheader("Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.2%}"
)

col2.metric(
    "Precision",
    f"{metrics['precision']:.2%}"
)

col3.metric(
    "Recall",
    f"{metrics['recall']:.2%}"
)

col4.metric(
    "F1 Score",
    f"{metrics['f1']:.2%}"
)

col5.metric(
    "ROC-AUC",
    f"{metrics['roc_auc']:.2%}"
)

st.divider()

# -----------------------------
# EMAIL INPUT
# -----------------------------

st.subheader("Test an Email")

email_text = st.text_area(
    "Paste Email Content Here",
    height=250
)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:

        cleaned_text = clean_text(email_text)

        vector = tfidf.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            vector
        )[0]

        confidence = (
            model.predict_proba(vector)
            .max() * 100
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"PHISHING EMAIL\n\nConfidence: {confidence:.2f}%"
            )

        else:

            st.success(
                f"LEGITIMATE EMAIL\n\nConfidence: {confidence:.2f}%"
            )

        st.progress(
            float(confidence / 100)
        )

st.divider()
