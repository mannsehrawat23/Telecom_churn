import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert to binary

# Preprocess the dataset
X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), drop_first=True)
y = df['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure the model directory exists
os.makedirs('model', exist_ok=True)

# Save the model
joblib.dump(model, 'model/Telecom_model.pkl')  # Fixed typo in filename
