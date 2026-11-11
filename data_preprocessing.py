
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate messages
df.drop_duplicates(inplace=True)

# Display dataset information
print("Dataset cleaned successfully!")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
print(df["label"].value_counts())

# Save cleaned dataset
df.to_csv("dataset/cleaned_spam.csv", index=False)

print("\nCleaned dataset saved successfully!")
