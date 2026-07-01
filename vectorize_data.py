import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("data/cleaned_news.csv")

# Check missing values
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna(subset=["content"])

# Convert everything to string
df["content"] = df["content"].astype(str)

X = df["content"]
y = df["label"]

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,   # keep top 5000 words
    stop_words="english"
)

# Convert text → numbers
X_vectorized = tfidf.fit_transform(X)

print("TF-IDF shape:", X_vectorized.shape)

# Save vectorizer for later use
import joblib
joblib.dump(tfidf, "model/vectorizer.pkl")

print("Vectorizer saved successfully!")