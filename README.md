\# SMS Spam Detection



\## Project Overview

This project is a Machine Learning-based SMS Spam Detection system that classifies SMS messages as Spam or Ham (Not Spam).



It uses Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier to identify spam messages.



\## Dataset

\- Dataset: SMS Spam Collection Dataset

\- Total cleaned messages: 5,169

\- Ham messages: 4,516

\- Spam messages: 653



\## Technologies Used

\- Python

\- Pandas

\- Scikit-learn

\- TF-IDF Vectorization

\- Multinomial Naive Bayes

\- Streamlit

\- Joblib



\## Model Performance

\- Accuracy: 96.62%

\- Precision (Spam): 99%

\- Recall (Spam): 74%

\- F1-score (Spam): 85%



\## Project Features

\- SMS message classification

\- Spam and Ham detection

\- Machine Learning model training

\- Interactive Streamlit web application

\- Real-time message prediction



\## Project Structure



```text

Spam\_SMS\_Detection/

│

├── dataset/

│   └── spam.csv

├── train.py

├── predict.py

├── app.py

├── spam\_model.pkl

├── vectorizer.pkl

├── requirements.txt

└── README.md

```



\## How to Run



\### 1. Install dependencies



```bash

pip install -r requirements.txt

```



\### 2. Train the model



```bash

python train.py

```



\### 3. Run the prediction program



```bash

python predict.py

```



\### 4. Launch the web application



```bash

streamlit run app.py

```



\## Conclusion



The SMS Spam Detection project demonstrates how Machine Learning and Natural Language Processing can be used to identify unwanted SMS messages.



The trained model achieved 96.62% accuracy on the test dataset.



\---



Developed as part of the CodSoft Machine Learning Internship.

