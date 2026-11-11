
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_spam.csv")

# Dataset information
print("Dataset Overview:")
print(df.info())

# Count spam and ham messages
print("\nMessage Distribution:")
print(df["label"].value_counts())

# Set visualization style
sns.set_style("whitegrid")

# 1. Spam vs Ham Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")
plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.savefig("dataset/spam_ham_distribution.png")
plt.show()

# 2. Message Length Distribution
df["message_length"] = df["message"].apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="message_length",
    hue="label",
    bins=50,
    kde=True
)
plt.title("SMS Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.savefig("dataset/message_length_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Graphs saved inside the dataset folder.")
