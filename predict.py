
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("\n================================")
print("       SMS SPAM DETECTOR")
print("================================")
print("Enter an SMS message to check.")
print("Type 'exit' to close the program.")

while True:
    message = input("\nEnter SMS: ")

    if message.lower() == "exit":
        print("Thank you for using SMS Spam Detector!")
        break

    if not message.strip():
        print("Please enter a valid message.")
        continue

    # Convert message into numerical features
    message_tfidf = vectorizer.transform([message])

    # Predict message category
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        print("\nResult: SPAM MESSAGE!")
    else:
        print("\nResult: NOT SPAM (HAM)")
