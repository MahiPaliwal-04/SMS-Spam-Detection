
import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Application title
st.title("📩 SMS Spam Detector")
st.write("Check whether an SMS message is Spam or Not Spam.")

st.divider()

# User input
message = st.text_area(
    "Enter your SMS message below:",
    placeholder="Type your message here..."
)

# Prediction button
if st.button("Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE DETECTED!")
        else:
            st.success("✅ THIS IS A NORMAL MESSAGE (HAM)")

st.divider()

st.caption("Machine Learning Model: Multinomial Naive Bayes")
st.caption("Text Feature Extraction: TF-IDF")
st.caption("Developed using Python and Streamlit")
