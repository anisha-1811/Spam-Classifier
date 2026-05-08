# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, render_template
# pyrefly: ignore [missing-import]
import joblib

app = Flask(__name__)

# Load model once at startup
model      = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data    = request.get_json()
    message = data.get("message", "")
    vec     = vectorizer.transform([message])
    result  = model.predict(vec)[0]          # "spam" or "ham"
    proba   = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 2)
    return jsonify({"result": result, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)