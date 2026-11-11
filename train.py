
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Step 2: Clean dataset
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df = df.dropna()
df = df.drop_duplicates()

df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Step 3: Separate features and labels
X = df["message"]
y = df["label"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train machine learning model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== SMS SPAM DETECTOR ==========")
print("\nModel Training Completed Successfully!")
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Ham", "Spam"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")
print("Files created: spam_model.pkl and vectorizer.pkl")
