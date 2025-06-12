import streamlit as st
import joblib

# Load model and vectorizer
vec = joblib.load("vec_tfidf.pkl")
model = joblib.load("xgb_spam_detector.pkl")

# Prediction function
def predict_spam(text):
    x = vec.transform([text])
    pred = model.predict(x)[0]
    return "🚨 Scam Message Detected" if pred == 1 else "✅ Safe Message"

# Page config
st.set_page_config(page_title="Scam Message Detector", page_icon="📩")

# Header section
st.markdown(
    """
    <div style="background-color:#f9f9f9;padding:20px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:#333333;text-align:center;">📩 Scam Message Detector</h1>
        <p style="text-align:center;font-size:16px;color:#666666;">
            Paste any suspicious message below and let the ML model decide if it's safe or a scam. Built with love using XGBoost 💻❤️
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input
user_input = st.text_area("✉️ Paste your message here:", height=150, placeholder="e.g. You've won a free trip to Dubai! Click now...")

# Predict button
if st.button("🧠 Analyze"):
    if user_input.strip() == "":
        st.warning("Oops! Message can't be empty.")
    else:
        result = predict_spam(user_input)
        st.markdown(
            f"""
            <div style="background-color:#ffffff;padding:15px 25px;border-radius:10px;border-left:6px solid #4CAF50;margin-top:20px;">
                <h3 style="color:#333333;">🔍 Prediction Result:</h3>
                <p style="font-size:18px;font-weight:bold;color:#222222;">{result}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    "<hr><p style='text-align:center;font-size:14px;'>Made by Annu ✨ | Powered by XGBoost + TF-IDF</p>",
    unsafe_allow_html=True
)
