# ================================
# 🍽️ Zomato Review Sentiment App
# ================================

import streamlit as st
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
os.chdir("C:/Zomato Data")

# ============================
# 📦 Download NLTK Resources
# ============================
nltk.download('stopwords')
nltk.download('wordnet')

# ============================
# 📦 Load Model and Vectorizer
# ============================
model = joblib.load("zomato_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ============================
# 🧠 Preprocessing Function
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    tokens = text.lower().split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# ============================
# 🎯 Streamlit UI
# ============================
st.set_page_config(page_title="Zomato Sentiment Analyzer", page_icon="🍽️")
st.title("🍽️ Zomato Review Sentiment Analyzer")
st.markdown("**Enter a Zomato-style review to see if it's Positive or Negative.**")

# User input
user_input = st.text_area("📝 Enter Review Here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        st.success(f"✅ Sentiment: **{prediction}**")
