import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load test dataset
df_test = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Separate features and target
X_test = df_test.drop(['customerID', 'Churn'], axis=1)
y_test = df_test['Churn'].map({'Yes':1, 'No':0})

# Load saved pipeline
pipeline = joblib.load('model/telecom_pipeline.pkl')

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model evaluation complete. Accuracy: {accuracy:.2f}")
