import streamlit as st
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
st.set_page_config(page_title="SpamShield AI", page_icon="📧")
st.title("📧 SpamShield AI - Email Spam Detector")
st.write("Detect whether an email message is Spam or Not using Machine Learning.")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
@st.cache_data
def load_dataset():
    data = pd.read_csv("spam.csv")
    return data

data = load_dataset()

# ---------------------------------------------------
# Text Cleaning Function (Custom Preprocessing)
# ---------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["cleaned_message"] = data["message"].apply(clean_text)

# ---------------------------------------------------
# Model Training
# ---------------------------------------------------
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["cleaned_message"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model(data)

# ---------------------------------------------------
# Sidebar Info
# ---------------------------------------------------
st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Model Accuracy: {round(accuracy*100, 2)}%")

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------
st.subheader("✉️ Enter an Email Message")

user_input = st.text_area("Type your message here")

if st.button("Check Spam"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed = clean_text(user_input)
        vectorized_input = vectorizer.transform([processed])
        prediction = model.predict(vectorized_input)[0]
        probability = model.predict_proba(vectorized_input).max()

        if prediction == "spam":
            st.error(f"⚠️ This message is SPAM ({round(probability*100,2)}% confidence)")
        else:
            st.success(f"✅ This message is NOT Spam ({round(probability*100,2)}% confidence)")