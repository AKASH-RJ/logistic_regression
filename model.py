import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("ds.csv")

# Features & target
X = df[['income', 'credit_score', 'loan_amount', 'years_with_bank']]
y = df['label'].map({'Approved': 1, 'Rejected': 0})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# Save model & scaler
joblib.dump(model, "loan_knn_model.pkl")
joblib.dump(scaler, "loan_scaler.pkl")

print("Model and scaler saved successfully!")
