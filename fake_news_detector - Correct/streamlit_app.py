import streamlit as st
import joblib

# Title
st.title("📰 Fake News Detector")
st.write("Detect whether a news article is **Fake** or **Real** using Machine Learning.")

# Load saved model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model/pac_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Input form
st.subheader("📩 Enter News Content:")
user_input = st.text_area("Paste the news article below", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        
        if prediction == "Fake":
            st.error("❌ This news is likely **Fake**.")
        else:
            st.success("✅ This news is likely **Real**.")
