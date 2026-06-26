import joblib

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Take input from user
news = input("Enter a news article:\n")

# Convert text to numbers
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

# Show result
if prediction[0] == 0:
    print("\nPrediction: FAKE NEWS ❌")
else:
    print("\nPrediction: REAL NEWS ✅")