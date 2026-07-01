import streamlit as st
import joblib
from PIL import Image

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# ---------------- Sidebar ----------------
st.sidebar.title("📰 Fake News Detector")
st.sidebar.write("Machine Learning + NLP Project")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Accuracy: 98.75%")

# ---------------- Main Page ----------------
st.title("📰 Fake News Detection System")
st.write("Enter a news article and click Predict.")

# Input box
news = st.text_area("Enter News Article")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Prediction ----------------
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:
        # Convert text
        news_vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vector)

        # Confidence
        confidence = model.predict_proba(news_vector)
        score = max(confidence[0]) * 100

        # Result
        if prediction[0] == 0:
            result = "❌ FAKE NEWS"
            st.error(result)
        else:
            result = "✅ REAL NEWS"
            st.success(result)

        # Save history
        st.session_state.history.append(result)

        # Confidence score
        st.subheader("📈 Confidence Score")
        st.write(f"Confidence: {score:.2f}%")
        st.progress(int(score))

        # News statistics
        st.subheader("📊 News Statistics")

        word_count = len(news.split())
        char_count = len(news)
        read_time = max(1, word_count // 200)

        st.write(f"📝 Words: {word_count}")
        st.write(f"🔤 Characters: {char_count}")
        st.write(f"⏱ Estimated Reading Time: {read_time} minute(s)")

# ---------------- Prediction History ----------------
st.subheader("📜 Prediction History")

if len(st.session_state.history) == 0:
    st.write("No predictions yet.")
else:
    for i, item in enumerate(st.session_state.history, 1):
        st.write(f"{i}. {item}")

# ---------------- Visualizations ----------------
st.subheader("📊 Dataset Visualization")

try:
    chart = Image.open("fake_real_chart.png")
    st.image(chart, caption="Fake vs Real News Distribution")

    wordcloud = Image.open("wordcloud.png")
    st.image(wordcloud, caption="Most Frequent Words")

except:
    st.warning(
        "Visualization images not found. "
        "Run: python visualize.py"
    )