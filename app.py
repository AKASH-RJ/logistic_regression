from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("loan_knn_model.pkl")
scaler = joblib.load("loan_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            income = float(request.form['income'])
            credit_score = float(request.form['credit_score'])
            loan_amount = float(request.form['loan_amount'])
            years_with_bank = float(request.form['years_with_bank'])

            features = np.array([[income, credit_score, loan_amount, years_with_bank]])
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            prediction = "Approved" if pred == 1 else "Rejected"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
