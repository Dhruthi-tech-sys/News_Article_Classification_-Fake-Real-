import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("Fake News Sample:")
print(fake.head())

print("\nTrue News Sample:")
print(true.head())

print("\nFake shape:", fake.shape)
print("True shape:", true.shape)