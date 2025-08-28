import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("C:\ml-ci-cd-pipeline2\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess the dataset
X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis = 1), drop_first = 1)
y = df['Churn']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101)


# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/Telecome_model.pkl')