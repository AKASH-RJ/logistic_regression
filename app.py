from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("logistic_regression.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Gender", "Married", "Loan_Status"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
X = df[["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]]
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    married = request.form['married']
    applicant_income = int(request.form['applicant_income'])
    loan_amount = int(request.form['loan_amount'])
    credit_history = int(request.form['credit_history'])

    # Encode input
    gender = label_encoders["Gender"].transform([gender])[0]
    married = label_encoders["Married"].transform([married])[0]

    # Predict
    features = np.array([[gender, married, applicant_income, loan_amount, credit_history]])
    prediction = model.predict(features)[0]
    prediction_label = label_encoders["Loan_Status"].inverse_transform([prediction])[0]

    return render_template("index.html", prediction_text=f"Loan Status: {prediction_label}")

if __name__ == "__main__":
    app.run(debug=True)
