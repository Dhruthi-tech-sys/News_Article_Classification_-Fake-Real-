import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("data/cleaned_news.csv")

# ---------- Bar Chart ----------
counts = df["label"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(["Fake", "Real"], [counts[0], counts[1]])
plt.title("Fake vs Real News Count")
plt.xlabel("News Type")
plt.ylabel("Count")

plt.savefig("fake_real_chart.png")
plt.close()

# ---------- Word Cloud ----------
text = " ".join(df["content"].dropna().astype(str).head(1000))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")

plt.savefig("wordcloud.png")
plt.close()

print("Charts saved successfully!")