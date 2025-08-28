import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Preprocess the dataset
X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), drop_first=True)
y = df['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Load model and feature names
model = joblib.load('model/Telecom_model.pkl')
feature_names = joblib.load('model/Telecom_model.pkl')

# Align test features with training features
X_test = X_test.reindex(columns=feature_names, fill_value=0)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
