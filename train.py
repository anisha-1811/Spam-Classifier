import pandas as pd
import requests
import zipfile
import io
import os
# pyrefly: ignore [missing-import]
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Download Dataset ---
print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data/")

# --- 2. Load & Prepare Data ---
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=["label", "message"])
print(f"Dataset loaded: {len(df)} messages")

X = df["message"]
y = df["label"]  # "spam" or "ham"

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Vectorize Text ---
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# --- 5. Train Model ---
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- 6. Evaluate ---
predictions = model.predict(X_test_vec)
print(f"✅ Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# --- 7. Save Model & Vectorizer ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Model saved to model/")