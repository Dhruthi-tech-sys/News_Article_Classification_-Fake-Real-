import pandas as pd
from utils.preprocess import clean_text

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
df = pd.concat([fake, true], axis=0)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

# Clean text
df["content"] = df["content"].apply(clean_text)

# Keep only required columns
df = df[["content", "label"]]

# Save final dataset
df.to_csv("data/cleaned_news.csv", index=False)

print("✅ Dataset prepared successfully!")
print(df.head())