

import joblib

# Load trained model
model = joblib.load("models/spam_classifier.pkl")

print("=" * 50)
print("       SMS SPAM DETECTION SYSTEM")
print("=" * 50)

while True:

    message = input("\nEnter an SMS message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        print("\nThank you for using SMS Spam Detection!")
        break

    if not message.strip():
        print("Please enter a valid message.")
        continue

    # Predict message category
    prediction = model.predict([message])[0]

    # Display result
    print("\nPrediction Result:")

    if prediction == "spam":
        print("WARNING! This message is SPAM.")
    else:
        print("This message is HAM (Not Spam).")

    # Display prediction confidence
    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities) * 100

    print("Confidence:", round(confidence, 2), "%")

    print("-" * 50)

