import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate (optional printout)
print(classification_report(y_test, model.predict(X_test)))

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature columns
with open("model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model trained and saved successfully.")