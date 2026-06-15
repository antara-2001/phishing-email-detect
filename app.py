from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# NLTK
# -----------------------------

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("phishing_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# METRICS
# -----------------------------

ACCURACY = 98.39
PRECISION = 98.20
RECALL = 98.72
F1 = 98.46
ROC_AUC = 99.83

# -----------------------------
# TEXT CLEANING
# -----------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# DASH APP
# -----------------------------

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG
    ]
)

# -----------------------------
# KPI CARDS
# -----------------------------

def metric_card(title, value):

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title),
                html.H3(value)
            ]
        )
    )

# -----------------------------
# RADAR CHART
# -----------------------------

radar = go.Figure()

radar.add_trace(
    go.Scatterpolar(
        r=[
            ACCURACY,
            PRECISION,
            RECALL,
            F1,
            ROC_AUC
        ],
        theta=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ],
        fill="toself"
    )
)

radar.update_layout(
    title="Model Performance"
)

# -----------------------------
# DONUT CHART
# -----------------------------

donut = px.pie(
    names=[
        "Legitimate",
        "Phishing"
    ],
    values=[
        41243,
        41243
    ],
    hole=0.6
)

# -----------------------------
# LAYOUT
# -----------------------------

app.layout = dbc.Container(
    [

        html.H1(
            "AI Phishing Email Detection Dashboard",
            className="text-center my-4"
        ),

        dbc.Row(
            [
                dbc.Col(
                    metric_card(
                        "Accuracy",
                        "98.39%"
                    )
                ),

                dbc.Col(
                    metric_card(
                        "Precision",
                        "98.20%"
                    )
                ),

                dbc.Col(
                    metric_card(
                        "Recall",
                        "98.72%"
                    )
                ),

                dbc.Col(
                    metric_card(
                        "F1",
                        "98.46%"
                    )
                ),

                dbc.Col(
                    metric_card(
                        "ROC-AUC",
                        "99.83%"
                    )
                ),
            ]
        ),

        html.Br(),

        dbc.Row(
            [

                dbc.Col(
                    dcc.Graph(
                        figure=radar
                    ),
                    width=6
                ),

                dbc.Col(
                    dcc.Graph(
                        figure=donut
                    ),
                    width=6
                ),
            ]
        ),

        html.Hr(),

        html.H3(
            "Email Scanner"
        ),

        dcc.Textarea(
            id="email-input",
            style={
                "width":"100%",
                "height":"200px"
            }
        ),

        html.Br(),

        dbc.Button(
            "Analyze Email",
            id="analyze-btn",
            color="danger"
        ),

        html.Br(),
        html.Br(),

        html.Div(
            id="prediction-output"
        ),

        html.Br(),

        dcc.Graph(
            id="risk-gauge"
        ),

        html.Hr(),

        html.H4(
            "Suspicious Keywords"
        ),

        html.Div(
            id="keywords-output"
        )
    ],
    fluid=True
)

# -----------------------------
# CALLBACK
# -----------------------------

@app.callback(
    [
        Output(
            "prediction-output",
            "children"
        ),

        Output(
            "risk-gauge",
            "figure"
        ),

        Output(
            "keywords-output",
            "children"
        )
    ],

    Input(
        "analyze-btn",
        "n_clicks"
    ),

    State(
        "email-input",
        "value"
    ),

    prevent_initial_call=True
)

def predict_email(
    n_clicks,
    email
):

    cleaned = clean_text(email)

    vector = tfidf.transform(
        [cleaned]
    )

    pred = model.predict(
        vector
    )[0]

    prob = (
        model.predict_proba(
            vector
        )[0][1] * 100
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            title={
                "text":
                "Phishing Risk (%)"
            }
        )
    )

    keywords = [
        "verify",
        "account",
        "password",
        "click",
        "urgent",
        "bank",
        "security",
        "login"
    ]

    found = [
        k for k in keywords
        if k in cleaned
    ]

    if pred == 1:

        result = dbc.Alert(
            f"PHISHING EMAIL ({prob:.2f}%)",
            color="danger"
        )

    else:

        result = dbc.Alert(
            f"LEGITIMATE EMAIL ({100-prob:.2f}%)",
            color="success"
        )

    return (
        result,
        gauge,
        ", ".join(found)
        if found else "None"
    )

# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
